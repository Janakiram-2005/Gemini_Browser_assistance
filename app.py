import os
import asyncio
import threading
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from pymongo import MongoClient
from bson.objectid import ObjectId

# 1. Load Secrets
load_dotenv()

app = Flask(__name__)
CORS(app) 

# --- GLOBAL KEY MANAGEMENT ---
ACTIVE_API_KEY = os.getenv("GOOGLE_API_KEY")
ACTIVE_MONGO_URI = os.getenv("MONGODB_URI")

# --- DATABASE SETUP ---
history_col = None
try:
    if ACTIVE_MONGO_URI:
        mongo_client = MongoClient(ACTIVE_MONGO_URI)
        db = mongo_client["gemini_agent_db"]
        history_col = db["history"]
        print(f"✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB Failed: {e}")

# --- GLOBAL VARIABLES ---
global_browser = None
global_context = None 
global_vision_state = False 
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

# --- MAIN AGENT ROUTE (FIXED) ---
@app.route('/run_agent', methods=['POST'])
def run_agent():
    data = request.json
    user_command = data.get('command')
    use_vision = data.get('use_vision', False)

    if not user_command: return jsonify({"status": "error", "message": "No command"}), 400
    if not ACTIVE_API_KEY: return jsonify({"status": "error", "message": "Server has no API Key!"}), 400

    print(f"🚀 Task: {user_command} | Vision Mode: {use_vision}")

    log_id = None
    if history_col is not None:
        log_id = history_col.insert_one({
            "command": user_command,
            "status": "Running",
            "result": "Processing...",
            "actions": [],
            "timestamp": datetime.now()
        }).inserted_id

    async def worker():
        global global_browser, global_context, current_async_task, global_vision_state
        current_async_task = asyncio.current_task()
        
        async def get_context():
            global global_browser, global_context, global_vision_state
            
            # 1. Start Browser
            if global_browser is None:
                print("🌐 Initializing Browser...")
                global_browser = Browser(config=BrowserConfig(headless=False))
            
            # 2. Start Context (if None or Vision changed)
            if global_context is None or global_vision_state != use_vision:
                print("🪟 Initializing Context...")
                if global_context: 
                    try: await global_context.close()
                    except: pass
                
                global_vision_state = use_vision
                global_context = await global_browser.new_context(
                    config=BrowserContextConfig(highlight_elements=use_vision)
                )
            
            # NOTE: We DO NOT call .new_page() here because it crashes. 
            # We let the Agent handle page creation.
            return global_context

        try:
            # 1. Get Context
            ctx = await get_context()
            
            dynamic_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                temperature=0,
                google_api_key=ACTIVE_API_KEY 
            )

            # 2. Run Agent (With Retry for "Closed Tab" error)
            try:
                agent = Agent(task=user_command, llm=dynamic_llm, browser_context=ctx)
                history_result = await agent.run()
            
            except Exception as e:
                error_str = str(e)
                # If error suggests the tab/window is closed/dead...
                if "Target closed" in error_str or "no valid pages" in error_str or "closed" in error_str:
                    print("⚠️ Context appears dead (Tab closed?). Retrying with fresh context...")
                    
                    # Force Reset
                    global_context = None 
                    ctx = await get_context() # Get fresh context
                    
                    # Retry
                    agent = Agent(task=user_command, llm=dynamic_llm, browser_context=ctx)
                    history_result = await agent.run()
                else:
                    raise e # Real error

            # 3. Process Results
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
                history_col.update_one({"_id": log_id}, {"$set": {
                    "status": "Stopped", "result": "Manually stopped.",
                }})
            return {"result": "Stopped", "actions": []}
        
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error: {error_msg}")
            
            status_code = "Failed"
            user_msg = "Error occurred."
            if "429" in error_msg or "Resource has been exhausted" in error_msg:
                status_code = "QuotaExceeded"
                user_msg = "⚠️ API Quota Exceeded."
            
            if log_id and history_col is not None:
                history_col.update_one({"_id": log_id}, {"$set": {
                    "status": status_code, "result": user_msg, "error_details": error_msg
                }})
            return {"result": user_msg, "actions": [], "status": status_code}

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
    app.run(debug=True, port=5000, threaded=False, use_reloader=False)