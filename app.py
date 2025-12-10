import os
import threading
import json
import asyncio
import base64
from datetime import datetime
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
        # connect=False is critical for Gunicorn workers to avoid deadlocks
        mongo_client = MongoClient(ACTIVE_MONGO_URI, connect=False)
        db = mongo_client["gemini_agent_db"]
        history_col = db["history"]
        print(f"✅ Database Configured")
except Exception as e:
    print(f"❌ MongoDB Configuration Error: {e}")

# --- GLOBAL VARIABLES ---
global_browser = None
global_context = None 
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
    try:
        history_col.delete_one({'_id': ObjectId(item_id)})
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/history', methods=['DELETE'])
def clear_history():
    if history_col is None: return jsonify({"status": "error"})
    try:
        history_col.delete_many({})
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop', methods=['POST'])
def stop_agent():
    async def force_stop():
        global current_async_task, global_context
        # 1. Cancel Python Task
        if current_async_task and not current_async_task.done():
            current_async_task.cancel()
        # 2. Kill Context (Tab)
        if global_context:
            try: await global_context.close()
            except: pass
            global_context = None 
    
    if agent_loop.is_running():
        agent_loop.call_soon_threadsafe(lambda: asyncio.create_task(force_stop()))
        return jsonify({"status": "stopped"})
    return jsonify({"status": "error"})

# --- MAIN AGENT ROUTE ---
@app.route('/run_agent', methods=['POST'])
def run_agent():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from browser_use import Agent, Browser, BrowserConfig
    from browser_use.browser.context import BrowserContextConfig

    data = request.json
    user_command = data.get('command')
    use_vision = data.get('use_vision', False)
    target_language = data.get('language', 'English')
    selected_model = data.get('model', 'gemini-2.0-flash') # Default to Flash

    if not user_command: return jsonify({"status": "error", "message": "No command"}), 400
    if not ACTIVE_API_KEY: return jsonify({"status": "error", "message": "Server has no API Key!"}), 400

    print(f"🚀 Task: {user_command} | Model: {selected_model} | Lang: {target_language}")

    # Prompt Engineering for Language
    if target_language and target_language.lower() != 'english':
        user_command += f" (IMPORTANT: Perform the task, but reply to me in {target_language} language only.)"

    log_id = None
    if history_col is not None:
        try:
            log_id = history_col.insert_one({
                "command": user_command,
                "status": "Running",
                "result": "Processing...",
                "timestamp": datetime.now()
            }).inserted_id
        except: pass

    async def worker():
        global global_browser, global_context, current_async_task
        current_async_task = asyncio.current_task()
        
        async def get_browser_and_context():
            global global_browser, global_context
            is_production = os.environ.get('RENDER') is not None

            # 1. Check Browser Health
            if global_browser:
                try:
                    if not global_browser.browser.is_connected():
                        global_browser = None
                        global_context = None
                except:
                    global_browser = None
                    global_context = None

            # 2. Init Browser
            if global_browser is None:
                if is_production and ACTIVE_BROWSERLESS_KEY:
                    global_browser = Browser(config=BrowserConfig(cdp_url=f"wss://chrome.browserless.io?token={ACTIVE_BROWSERLESS_KEY}"))
                else:
                    extra_args = ["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox", "--single-process"] if is_production else []
                    global_browser = Browser(config=BrowserConfig(headless=is_production, extra_chromium_args=extra_args))

            # 3. Reuse or Create Context
            if global_context:
                try: 
                    # Quick check if context is still valid
                    await global_context.pages() 
                except: 
                    global_context = None

            if global_context is None:
                global_context = await global_browser.new_context(config=BrowserContextConfig(highlight_elements=use_vision))

            return global_context

        try:
            ctx = await get_browser_and_context()

            # Pass the selected model here
            dynamic_llm = ChatGoogleGenerativeAI(
                model=selected_model, 
                temperature=0,
                google_api_key=ACTIVE_API_KEY 
            )

            agent = Agent(task=user_command, llm=dynamic_llm, browser_context=ctx)
            
            try:
                history_result = await agent.run()
            except Exception as e:
                print(f"⚠️ Agent Error: {e}. Resetting context...")
                global_context = None 
                raise e

            # 📸 CAPTURE SCREENSHOT (Visual Feedback)
            screenshot_b64 = None
            try:
                page = await ctx.pages()[0] # Get active page
                screenshot_bytes = await page.screenshot()
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            except: pass

            final_result = str(history_result.final_result()) if history_result.final_result() else "Task Completed."
            
            # Save to DB
            if log_id and history_col is not None:
                history_col.update_one({"_id": log_id}, {"$set": {"status": "Success", "result": final_result}})
            
            return {
                "status": "success", 
                "message": final_result, 
                "screenshot": screenshot_b64 
            }

        except asyncio.CancelledError:
            if log_id and history_col is not None:
                history_col.update_one({"_id": log_id}, {"$set": {"status": "Stopped", "result": "Manually stopped."}})
            return {"status": "error", "message": "Stopped by user"}
        
        except Exception as e:
            global_browser = None
            global_context = None
            error_msg = str(e)
            if log_id and history_col is not None:
                history_col.update_one({"_id": log_id}, {"$set": {"status": "Failed", "result": "Error", "error_details": error_msg}})
            return {"status": "error", "message": str(e)}

    try:
        future = asyncio.run_coroutine_threadsafe(worker(), agent_loop)
        result_data = future.result()
        return jsonify(result_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=False, use_reloader=False)