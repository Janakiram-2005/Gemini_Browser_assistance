import os
import threading
import json
import asyncio
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId

# 1. Load Secrets
load_dotenv()

app = Flask(__name__)
CORS(app) 

# --- GLOBAL KEY MANAGEMENT ---
ACTIVE_API_KEY = os.getenv("GOOGLE_API_KEY")
ACTIVE_MONGO_URI = os.getenv("MONGODB_URI")
ACTIVE_BROWSERLESS_KEY = os.getenv("BROWSERLESS_API_KEY")

# --- DATABASE SETUP ---
history_col = None
try:
    if ACTIVE_MONGO_URI:
        mongo_client = MongoClient(ACTIVE_MONGO_URI, connect=False)
        db = mongo_client["gemini_agent_db"]
        history_col = db["history"]
        print(f"✅ Database Configured")
except Exception as e:
    print(f"❌ MongoDB Configuration Error: {e}")

# --- GLOBAL VARIABLES ---
global_browser = None
global_context = None # We keep this now to reuse the open tab
current_async_task = None
agent_loop = asyncio.new_event_loop()

def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

t = threading.Thread(target=start_background_loop, args=(agent_loop,), daemon=True)
t.start()

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/update_key', methods=['POST'])
def update_server_key():
    global ACTIVE_API_KEY
    data = request.json
    new_key = data.get('new_key')
    if new_key:
        ACTIVE_API_KEY = new_key
        return jsonify({"status": "success", "message": "Server API Key Updated!"})
    return jsonify({"status": "error", "message": "No key provided"}), 400

@app.route('/history', methods=['GET'])
def get_history():
    if history_col is None: return jsonify([])
    try:
        cursor = history_col.find().sort("timestamp", -1).limit(50)
        logs = []
        for doc in cursor:
            logs.append({
                "_id": str(doc.get("_id")),
                "command": doc.get("command"),
                "status": doc.get("status", "Unknown"),
                "result": doc.get("result", ""),
                "actions": doc.get("actions", []),
                "time": doc.get("timestamp").strftime("%H:%M:%S")
            })
        return jsonify(logs)
    except:
        return jsonify([])

@app.route('/usage', methods=['GET'])
def get_usage():
    if history_col is None: return jsonify({"used": 0, "limit": 50})
    try:
        start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        count = history_col.count_documents({"timestamp": {"$gte": start_of_day}})
        return jsonify({"used": count, "limit": 50})
    except:
        return jsonify({"used": 0, "limit": 50})

@app.route('/history/<item_id>', methods=['DELETE'])
def delete_history_item(item_id):
    if history_col is None: return jsonify({"status": "error"})
    history_col.delete_one({'_id': ObjectId(item_id)})
    return jsonify({"status": "success"})

@app.route('/history', methods=['DELETE'])
def clear_history():
    if history_col is None: return jsonify({"status": "error"})
    history_col.delete_many({})
    return jsonify({"status": "success"})

@app.route('/stop', methods=['POST'])
def stop_agent():
    def cancel_task():
        global current_async_task
        if current_async_task and not current_async_task.done():
            current_async_task.cancel()
            return True
        return False
    if agent_loop.is_running():
        agent_loop.call_soon_threadsafe(cancel_task)
        return jsonify({"status": "stopped"})
    return jsonify({"status": "error"})

# --- MAIN AGENT ROUTE (KEEP TAB OPEN) ---
@app.route('/run_agent', methods=['POST'])
def run_agent():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from browser_use import Agent, Browser, BrowserConfig
    from browser_use.browser.context import BrowserContextConfig

    data = request.json
    user_command = data.get('command')
    use_vision = data.get('use_vision', False)

    if not user_command: return jsonify({"status": "error", "message": "No command"}), 400
    if not ACTIVE_API_KEY: return jsonify({"status": "error", "message": "Server has no API Key!"}), 400

    print(f"🚀 Task: {user_command} | Vision Mode: {use_vision}")

    log_id = None
    if history_col is not None:
        try:
            log_id = history_col.insert_one({
                "command": user_command,
                "status": "Running",
                "result": "Processing...",
                "actions": [],
                "timestamp": datetime.now()
            }).inserted_id
        except Exception as e:
            print(f"⚠️ DB Insert Warning: {e}")

    async def worker():
        global global_browser, global_context, current_async_task
        current_async_task = asyncio.current_task()
        
        async def get_browser_and_context():
            global global_browser, global_context
            is_production = os.environ.get('RENDER') is not None

            # 1. Health Check: Is the browser alive?
            if global_browser:
                try:
                    if not global_browser.browser.is_connected():
                        print("💔 Dead browser detected. Resetting...")
                        global_browser = None
                        global_context = None
                except:
                    global_browser = None
                    global_context = None

            # 2. Initialize Browser (If needed)
            if global_browser is None:
                if is_production and ACTIVE_BROWSERLESS_KEY:
                    print(f"🌐 Connecting to Browserless...")
                    global_browser = Browser(
                        config=BrowserConfig(
                            cdp_url=f"wss://chrome.browserless.io?token={ACTIVE_BROWSERLESS_KEY}"
                        )
                    )
                else:
                    print(f"🌐 Launching Local Browser...")
                    extra_args = ["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox", "--single-process"] if is_production else []
                    global_browser = Browser(
                        config=BrowserConfig(
                            headless=is_production, # Change to False if you want to see it pop up on laptop
                            extra_chromium_args=extra_args
                        )
                    )

            # 3. Reuse or Create Context (Tab)
            # If we already have a tab open, reuse it!
            if global_context:
                try:
                    # Test if context is valid (hacky check)
                    # We assume it is valid if we have it, but if agent fails we reset
                    pass 
                except:
                     global_context = None

            if global_context is None:
                print("✨ Creating fresh context (New Tab)...")
                global_context = await global_browser.new_context(
                    config=BrowserContextConfig(highlight_elements=use_vision)
                )

            return global_context

        try:
            # 1. Get Context (Reuse existing if possible)
            ctx = await get_browser_and_context()

            dynamic_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                temperature=0,
                google_api_key=ACTIVE_API_KEY 
            )

            # 2. Run Agent
            agent = Agent(task=user_command, llm=dynamic_llm, browser_context=ctx)
            
            try:
                history_result = await agent.run()
            except Exception as e:
                # If agent crashes, it usually means the Tab is dead/stuck.
                print(f"⚠️ Agent Error: {e}. Resetting context for next run...")
                global_context = None # Force new tab next time
                raise e

            # 3. SUCCESS! We do NOT close the context here.
            print("✅ Task Done. Keeping tab open.")

            # Process Results
            actions_log = []
            final_result = str(history_result.final_result()) if history_result.final_result() else "Task Completed."
            
            try:
                for step in history_result.history:
                    if hasattr(step, 'tool_calls'):
                        for tool in step.tool_calls:
                            name = tool.function.name
                            if 'go_to' in name: text = "Opened website"
                            elif 'click' in name: text = "Clicked element"
                            elif 'input' in name: text = "Typed text"
                            else: text = "Performed action"
                            actions_log.append({"thought": text, "action": name})
            except: pass

            if log_id and history_col is not None:
                history_col.update_one({"_id": log_id}, {"$set": {
                    "status": "Success", "result": final_result, "actions": actions_log
                }})
            
            return {"result": final_result, "actions": actions_log}

        except asyncio.CancelledError:
            if log_id and history_col is not None:
                history_col.update_one({"_id": log_id}, {"$set": {"status": "Stopped", "result": "Manually stopped."}})
            return {"result": "Stopped", "actions": []}
        
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error: {error_msg}")
            
            # If error, force reset for next time
            global_browser = None
            global_context = None
            
            status_code = "Failed"
            if "429" in error_msg: status_code = "QuotaExceeded"
            
            if log_id and history_col is not None:
                history_col.update_one({"_id": log_id}, {"$set": {
                    "status": status_code, "result": "Error occurred.", "error_details": error_msg
                }})
            return {"result": "Error occurred.", "actions": [], "status": status_code}

    try:
        future = asyncio.run_coroutine_threadsafe(worker(), agent_loop)
        result_data = future.result()
        
        if result_data.get("status") == "QuotaExceeded":
             return jsonify({"status": "maintenance", "message": result_data.get("result")})

        return jsonify({
            "status": "success", 
            "message": result_data.get("result", ""),
            "actions": result_data.get("actions", [])
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=False, use_reloader=False)