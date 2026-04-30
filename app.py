import os
import sys
import json
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from google import genai
from google.genai import types

# --- 1. CONFIG & SECRETS ---
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# GOOGLE CLOUD CONFIG
VERTEX_SA_FILE = "convertionalai-d8da9e4d43dd.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VERTEX_SA_FILE

PROJECT_ID = "convertionalai"
LOCATION = "us-central1"

# INITIALIZE GENAI CLIENT (Vertex AI Mode)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# DATABASE (Optional for History)
history_col = None
try:
    mongo_uri = os.getenv("MONGODB_URI")
    if mongo_uri:
        mongo_client = MongoClient(mongo_uri, connect=False)
        db = mongo_client["gemini_agent_db"]
        history_col = db["history"]
        print("✅ [DB] Database Configured")
except Exception as e:
    print(f"❌ [DB] Error: {e}")

# LOGGING SYSTEM
LOG_QUEUE = []
MAX_CONTEXT = 6000
MAX_REPEAT = 2

def log(level, msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {"time": timestamp, "msg": f"[{level}] {msg}"}
    LOG_QUEUE.append(entry)
    if len(LOG_QUEUE) > 50: LOG_QUEUE.pop(0)
    print(f"[{timestamp}] [{level}] {msg}")
    sys.stdout.flush()

# ---------- TASK TYPE DETECTION ----------
def detect_task_type(command):
    if any(k in command.lower() for k in ["code", "leetcode", "submit", "function", "class", "write code"]):
        return "coding"
    return "normal"

# ---------- CODING PROMPT ----------
CODING_PROMPT = """
You are a coding automation agent.

Steps:
1. Generate correct working code
2. Paste into editor
3. Click submit
4. STOP

Return JSON action only.
"""

# ---------- SYSTEM PROMPT (AUTONOMOUS BRAIN) ----------
SYSTEM_PROMPT = """
SYSTEM ROLE: Autonomous Browser Execution Engine (BRAIN)

OBJECTIVE:
You are the planner and reasoning engine. You decide the best next step to achieve the user's goal.
The frontend is the EXECUTOR.

CRITICAL PROTOCOL:
1. ONLY use DOM text for reasoning.
2. DO NOT engage in conversation.
3. Your FIRST response must be a physical action (click, type, open_new_tab).
4. ONLY use "chat" when the goal is achieved or a direct question is asked.
5. Return ONE action in valid JSON.

AVAILABLE ACTIONS:
- open_new_tab: value="URL"
- close_tab: (no value)
- switch_tab: (no value)
- click: selector="CSS", value="Visible text fallback"
- type: selector="CSS", value="Text to input"
- scroll: value="up | down"
- wait: (no value)
- stop: (no value)
- chat: value="Response to user"

OUTPUT JSON:
{
 "action": "",
 "selector": "",
 "value": "",
 "caption": "User-friendly status",
 "reason": "Why this action?",
 "confidence": 0-1,
 "is_goal_achieved": false
}
"""

# ---------- AGENT ----------
@app.route("/agent", methods=["POST"])
def agent():
    try:
        data = request.json or {}
        command = data.get("command", "")
        instruction = data.get("instruction")  # OPTIONAL MODE
        context = data.get("context", "")[:MAX_CONTEXT]
        prev_context = data.get("previous_context", "")
        last_action = data.get("last_action", "")
        repeat_count = data.get("repeat_count", 0)

        log("INFO", f"Command: {command} | Mode: {instruction}")
        
        # PRIORITIZE SELECTED MODE
        task_type = instruction if instruction else detect_task_type(command)

        # PREPARE PROMPT
        mode_guidance = f"MODE: {instruction.upper()}" if instruction else "MODE: GENERAL"
        
        if task_type == "coding":
            prompt = f"{CODING_PROMPT}\nTASK: {command}\n{mode_guidance}\nPAGE: {context[:4000]}\nLAST ACTION: {last_action}"
        else:
            prompt = f"{SYSTEM_PROMPT}\nGOAL: {command}\n{mode_guidance}\nPAGE: {context[:4000]}\nPREVIOUS: {prev_context[:2000]}\nLAST ACTION: {last_action}"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"temperature": 0}
        )

        try:
            text = response.text.strip()
            if "```json" in text: text = text.split("```json")[1].split("```")[0].strip()
            result = json.loads(text)
        except Exception as e:
            log("ERROR", f"Parsing Error: {e}")
            result = {"action": "wait", "reason": "Parsing failed", "confidence": 0}

        # ENHANCE SELECTOR
        if not result.get("selector") and result.get("value") and result.get("action") in ["click", "type"]:
            result["fallback_selector"] = f"*:contains('{result['value']}')"

        # NAVIGATION SAFETY (LOOP PREVENTION)
        if result.get("action") in ["open_new_tab", "navigate"]:
            result["action"] = "open_new_tab" # Standardize
            # Only stop if it's a simple navigation task
            if any(k in command.lower() for k in ["open", "go to", "visit"]) and len(command.split()) < 4:
                result["is_goal_achieved"] = True

        # CODING MODE STATE MACHINE
        if task_type == "coding":
            code = result.get("value", "")
            # Step 1: Detect Code and Type it
            if last_action != "type" and any(k in code for k in ["class ", "def ", "function", "import "]):
                result["action"] = "type"
                result["selector"] = "textarea, .CodeMirror, [contenteditable='true'], .monaco-editor"
                result["caption"] = "Writing and pasting generated code..."
            # Step 2: Submit after Typing
            elif last_action == "type":
                result["action"] = "click"
                result["selector"] = "button[type='submit'], .submit-btn, button:contains('Submit')"
                result["caption"] = "Submitting your code solution..."
            # Step 3: Finish after Submitting
            elif last_action == "click":
                result["action"] = "stop"
                result["is_goal_achieved"] = True
                result["caption"] = "Coding task successfully completed!"

        # SAVE TO HISTORY
        if history_col is not None and command:
            try:
                history_col.insert_one({
                    "command": command,
                    "timestamp": datetime.now(),
                    "task_type": task_type,
                    "status": "success"
                })
            except: pass

        # LOW CONFIDENCE FALLBACK
        if result.get("confidence", 1) < 0.2:
            result["action"] = "wait"
            result["caption"] = "Thinking carefully... confidence is low."

        # RESPONSE
        final_res = {
            "action": result.get("action"),
            "selector": result.get("selector", ""),
            "fallback_selector": result.get("fallback_selector", ""),
            "value": result.get("value", ""),
            "caption": result.get("caption", result.get("reason", "Working...")),
            "reason": result.get("reason", ""),
            "confidence": result.get("confidence", 0.5),
            "is_goal_achieved": result.get("is_goal_achieved", False),
            "meta": {
                "repeat_count": repeat_count,
                "task": task_type,
                "time": datetime.now().isoformat()
            }
        }

        log("ACTION", f"{final_res['action']} (Conf: {final_res['confidence']})")
        return jsonify(final_res)

    except Exception as e:
        error_msg = str(e)
        log("CRITICAL", error_msg)
        # Return valid JSON even on error to prevent 500 status in frontend
        return jsonify({
            "action": "chat", 
            "value": f"System Error: {error_msg}", 
            "caption": "Brain Error: Backing off...",
            "is_goal_achieved": True # STOP the loop on critical failure
        })

# --- 4. WEB INTERFACE ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/logs", methods=["GET"])
def get_logs():
    return jsonify(LOG_QUEUE)

@app.route("/history", methods=["GET"])
def get_history():
    if history_col is None: return jsonify([])
    try:
        cursor = history_col.find().sort("timestamp", -1).limit(20)
        logs = [{"command": doc.get("command"), "time": doc.get("timestamp").strftime("%H:%M:%S")} for doc in cursor]
        return jsonify(logs)
    except: return jsonify([])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)