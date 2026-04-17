import os
import threading
import json
import asyncio
import base64
import time
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

# 1. Load Secrets
load_dotenv()

app = Flask(__name__)

# --- CRITICAL: ENABLE CORS FOR CHROME EXTENSION ---
CORS(app, resources={r"/*": {"origins": "*"}})

# --- GLOBAL KEY MANAGEMENT ---
ACTIVE_API_KEY = os.getenv("GOOGLE_API_KEY")
ACTIVE_MONGO_URI = os.getenv("MONGODB_URI")
ACTIVE_BROWSERLESS_KEY = os.getenv("BROWSERLESS_API_KEY")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Map deprecated model ids to a currently supported default.
MODEL_ALIASES = {
    "gemini-2.0-flash": "gemini-2.5-flash",
    "gemini-2.0-pro-exp": "gemini-2.5-flash",
}


def normalize_model_name(model_name):
    name = (model_name or "").strip()
    if not name:
        return DEFAULT_GEMINI_MODEL
    return MODEL_ALIASES.get(name, name)

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
global_context = None 
current_async_task = None
agent_loop = asyncio.new_event_loop()

def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

t = threading.Thread(target=start_background_loop, args=(agent_loop,), daemon=True)
t.start()

# --- HELPER: RETRY LOGIC FOR GOOGLE API ---
def retry_with_backoff(func, *args, **kwargs):
    import re
    retries = 10
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "ResourceExhausted" in error_str:
                wait = (1.5 ** i) + random.uniform(0, 1)
                
                # Check if API tells us exact retry delay
                match = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}', error_str)
                if match:
                    delay = int(match.group(1))
                    wait = max(delay + 1, wait)
                
                print(f"⚠️ 429 Rate Limit. Retrying in {wait:.2f}s... (Attempt {i+1}/{retries})")
                time.sleep(wait)
            else:
                raise e
    raise Exception("API Quota Exceeded after retries")

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

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
    except: return jsonify([])

@app.route('/history/<item_id>', methods=['DELETE'])
def delete_history_item(item_id):
    if history_col is None: return jsonify({"status": "error"})
    try:
        history_col.delete_one({'_id': ObjectId(item_id)})
        return jsonify({"status": "success"})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})

@app.route('/history', methods=['DELETE'])
def clear_history():
    if history_col is None: return jsonify({"status": "error"})
    try:
        history_col.delete_many({})
        return jsonify({"status": "success"})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})

# --- ROUTE 1: COPILOT EXTENSION (THE BRAIN) ---
@app.route('/analyze', methods=['POST'])
def analyze_page():
    data = request.json
    user_prompt = data.get('prompt', '')
    tabs_list = data.get('tabs', [])
    page_text = data.get('current_text', '')
    current_page_signature = data.get('current_page_signature', '')
    previous_page_signature = data.get('previous_page_signature', '')
    repeat_count = int(data.get('repeat_count', 0) or 0)
    
    # --- 1. MEMORY INJECTION (Read what we just did) ---
    last_action = data.get('last_action', 'None') 
    last_params = data.get('last_params', {})
    
    # --- 1b. HISTORY LOGGING (New Command) ---
    if last_action == "None" and history_col is not None and user_prompt:
        try:
            history_col.insert_one({
                "command": user_prompt,
                "status": "Running",
                "result": "Processing...",
                "timestamp": datetime.now()
            })
        except: pass

    # --- 2. CIRCUIT BREAKER (Stop Loop if Skill Finished) ---
    # If the frontend reports it just finished a skill, we stop here.
    if last_action in ["youtube_search", "whatsapp_send", "youtube_control"]:
        print(f"🛑 VERIFICATION COMPLETE: '{last_action}' finished. Stopping Agent.")
        return jsonify({
            "thought": f"I verified that {last_action} was successfully triggered. Task complete.",
            "action": "chat",
            "reply": "Task executed successfully.",
            "status": "completed" 
        })

    if not ACTIVE_API_KEY:
        return jsonify({"thought": "Error", "action": "chat", "reply": "No API Key"})

    llm = ChatGoogleGenerativeAI(
        model=normalize_model_name(DEFAULT_GEMINI_MODEL),
        temperature=0,
        google_api_key=ACTIVE_API_KEY,
        max_retries=1,
    )
    
    # --- VS CODE TERMINAL LOGGING ---
    print(f"\n🔵 USER GOAL: {user_prompt}")
    print(f"🔙 PREVIOUS ACTION: {last_action}")
    print(f"🔙 PREVIOUS PARAMS: {last_params}")
    print(f"🔁 REPEAT COUNT: {repeat_count}")

    # --- 3. SUPER PROMPT (With Memory & Verification Rules) ---
    system_instruction = f"""
    You are an Autonomous Browser Agent.
    
    USER GOAL: "{user_prompt}"
    LAST ACTION: "{last_action}"
    LAST PARAMS: {json.dumps(last_params)}
    OPEN TABS: {json.dumps(tabs_list)}
    CURRENT PAGE CONTEXT (Truncated): {page_text[:4000]}
    
    CAPABILITIES:
    1. **General Interaction:** "click_element", "type_text", "scroll"
    2. **Navigation:** "switch_tab", "open_url", "close_tabs"
    3. **Specialized Skills:**
       - "youtube_search" (Params: "query") - USAGE: ONLY if on YouTube or if user asks for video.
       - "whatsapp_send" (Params: "contact", "message")
    
    CRITICAL RULES (FOLLOW STRICTLY):
    1. **TERMINATION:** IF the "LAST ACTION" successfully completed the user's goal, YOU MUST STOP IMMEDIATELY.
       - SET "action": "chat", "status": "completed", "reply": "Done."
    2. **LOOP DETECTION:** 
       - If you just did "click_element" on "Spam" and it failed, DO NOT click "Spam" again. Try "More" or scroll.
         - Repeating the same action can be valid for quizzes/forms when the page state changes.
         - If "LAST ACTION" + "LAST PARAMS" are identical AND page state did not change, CHANGE STRATEGY.
    3. **MEMORY:** If you just opened a URL, you are likely ON that page. Do not open it again.
    
    RESPONSE JSON FORMAT (Strict):
    {{
      "thought": "Brief reasoning.",
      "action": "click_element" | "type_text" | "scroll" | "switch_tab" | "open_url" | "close_tabs" | "chat" | "youtube_search",
      "params": {{ "target_text": "..." }} OR {{ "url": "..." }} OR {{ "query": "..." }},
      "tab_id": 123,
      "status": "in_progress" | "completed"
    }}
    """
    
    import re 
    try:
        response = retry_with_backoff(llm.invoke, system_instruction)
        content = response.content.strip()
        
        # --- ROBUST JSON EXTRACTION ---
        try:
            # 1. Try finding a markdown block
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            # 2. If valid JSON chars exist, try to parse the first dict found
            # This handles "Okay, here is the JSON: { ... }"
            elif "{" in content:
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    content = match.group(0)

            res_json = json.loads(content)
            
            # --- LOOP DETECTION GUARDRAIL ---
            # Now we compare PARAMS too. 
            current_action = res_json.get('action')
            current_params = res_json.get('params', {})

            is_loop = False
            if current_action == last_action and last_action not in ["bg_scroll", "type_text", "None"]:
                same_params = current_params == last_params
                same_page = bool(current_page_signature) and current_page_signature == previous_page_signature

                # Only force-stop when the agent repeats the same step several times on unchanged page state.
                if same_params and same_page and repeat_count >= 2:
                    is_loop = True

            if is_loop:
                print(f"⚠️ LOOP DETECTED: Agent repeated '{last_action}' with unchanged page. Forcing STOP.")
                res_json = {
                    "thought": f"I detected a loop ({last_action} repeated). Stopping to avoid error.",
                    "action": "chat",
                    "status": "completed",
                    "reply": "I tried the same action multiple times on an unchanged page and it seems stuck. Please check the page manually."
                }
            
            print(f"🤔 THOUGHT: {res_json.get('thought')}")
            print(f"⚡ ACTION: {res_json.get('action')}")
            print(f"🔄 STATUS: {res_json.get('status')}")
            
            return json.dumps(res_json) # Return valid JSON string
            
        except:
            print(f"⚠️ RAW RESPONSE: {content}")

        return content
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return jsonify({"thought": "Error", "status": "completed", "action": "chat", "reply": f"AI Error: {str(e)}"})


@app.route('/stop', methods=['POST'])
def stop_agent():
    async def force_stop():
        global current_async_task, global_context
        if current_async_task and not current_async_task.done(): current_async_task.cancel()
        if global_context:
            try: await global_context.close()
            except: pass
            global_context = None 
    if agent_loop.is_running():
        agent_loop.call_soon_threadsafe(lambda: asyncio.create_task(force_stop()))
        return jsonify({"status": "stopped"})
    return jsonify({"status": "error"})


# --- ROUTE 2: AUTONOMOUS AGENT (THE HANDS) ---
@app.route('/run_agent', methods=['POST'])
def run_agent():
    from browser_use import Agent, Browser, BrowserConfig
    from browser_use.browser.context import BrowserContextConfig

    data = request.json
    user_command = data.get('command')
    use_vision = data.get('use_vision', False)
    selected_model = normalize_model_name(data.get('model', DEFAULT_GEMINI_MODEL))

    if not ACTIVE_API_KEY: return jsonify({"status": "error", "message": "No API Key"}), 400

    print(f"🚀 Autonomous Task: {user_command}")

    log_id = None
    if history_col is not None:
        try:
            log_id = history_col.insert_one({
                "command": user_command, "status": "Running", "result": "Processing...", "timestamp": datetime.now()
            }).inserted_id
        except: pass

    async def worker():
        global global_browser, global_context, current_async_task
        current_async_task = asyncio.current_task()
        
        try:
            if global_browser is None:
                global_browser = Browser(config=BrowserConfig(headless=False))
            
            if global_context is None:
                global_context = await global_browser.new_context(config=BrowserContextConfig(highlight_elements=use_vision))

            llm = ChatGoogleGenerativeAI(
                model=selected_model, 
                temperature=0,
                google_api_key=ACTIVE_API_KEY
            )

            agent = Agent(
                task=user_command, 
                llm=llm, 
                browser_context=global_context
            )
            
            history_result = await agent.run(max_steps=20)
            
            final_result = str(history_result.final_result()) if history_result.final_result() else "Task Completed."
            
            if log_id: history_col.update_one({"_id": log_id}, {"$set": {"status": "Success", "result": final_result}})
            return {"status": "success", "message": final_result}

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Agent Failed: {error_msg}")
            if log_id: history_col.update_one({"_id": log_id}, {"$set": {"status": "Failed", "result": error_msg}})
            return {"status": "error", "message": error_msg}

    try:
        future = asyncio.run_coroutine_threadsafe(worker(), agent_loop)
        return jsonify(future.result())
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)