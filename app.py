import os
import sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
import json
import requests
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient

# --- 1. CONFIG & SECRETS ---
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ──────────────────────────────────────────────
# OLLAMA CONFIG  (local LLM server)
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OLLAMA_MODELS = {
    "gemma2":   "gemma2:latest",   # local Gemini-family model
    "qwen2.5":  "qwen2.5:0.5b",   # local Qwen model (fast, lightweight)
}

# ──────────────────────────────────────────────
# GOOGLE VERTEX AI CONFIG  (cloud fallback)
# ──────────────────────────────────────────────
VERTEX_SA_FILE = "convertionalai-d8da9e4d43dd.json"
PROJECT_ID     = "convertionalai"
LOCATION       = "us-central1"

vertex_client = None
try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VERTEX_SA_FILE
    from google import genai
    vertex_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    print("[OK] [VERTEX] Cloud Gemini client ready")
except Exception as e:
    print(f"[WARN] [VERTEX] Skipped: {e}")

# ──────────────────────────────────────────────
# DATABASE (Optional)
# ──────────────────────────────────────────────
history_col = None
try:
    mongo_uri = os.getenv("MONGODB_URI")
    if mongo_uri:
        mongo_client = MongoClient(mongo_uri, connect=False)
        db          = mongo_client["gemini_agent_db"]
        history_col = db["history"]
        print("[OK] [DB] Database Configured")
except Exception as e:
    print(f"[ERR] [DB] Error: {e}")

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
LOG_QUEUE  = []
MAX_CONTEXT = 6000
MAX_REPEAT  = 2

def log(level, msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {"time": timestamp, "msg": f"[{level}] {msg}"}
    LOG_QUEUE.append(entry)
    if len(LOG_QUEUE) > 50:
        LOG_QUEUE.pop(0)
    print(f"[{timestamp}] [{level}] {msg}")
    sys.stdout.flush()

# ──────────────────────────────────────────────
# PROMPTS
# ──────────────────────────────────────────────
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

CODING_PROMPT = """
You are a coding automation agent.

Steps:
1. Generate correct working code
2. Paste into editor
3. Click submit
4. STOP

Return JSON action only.
"""

# ──────────────────────────────────────────────
# TASK DETECTION
# ──────────────────────────────────────────────
def detect_task_type(command):
    keywords = ["code", "leetcode", "submit", "function", "class", "write code"]
    if any(k in command.lower() for k in keywords):
        return "coding"
    return "normal"

# ──────────────────────────────────────────────
# LLM ROUTING LAYER
# ──────────────────────────────────────────────
def call_ollama(model_key: str, prompt: str) -> str:
    """Call a local Ollama model and return raw text response."""
    model_name = OLLAMA_MODELS.get(model_key, model_key)
    log("LLM", f"Using Ollama → {model_name}")

    payload = {
        "model":  model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def call_vertex(prompt: str) -> str:
    """Call cloud Gemini via Vertex AI and return raw text response."""
    if vertex_client is None:
        raise RuntimeError("Vertex AI client not initialized")
    log("LLM", "Using Cloud Gemini (Vertex AI)")
    response = vertex_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0}
    )
    return response.text.strip()


def generate_response(model_key: str, prompt: str) -> str:
    """Route the prompt to the correct LLM backend."""
    if model_key in OLLAMA_MODELS:
        return call_ollama(model_key, prompt)
    elif model_key == "gemini-cloud":
        return call_vertex(prompt)
    else:
        # Auto-fallback: try Ollama first, then Vertex
        try:
            return call_ollama("gemma2", prompt)
        except Exception:
            return call_vertex(prompt)

# ──────────────────────────────────────────────
# PARSE LLM OUTPUT → JSON
# ──────────────────────────────────────────────
def parse_llm_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)

# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def list_models():
    """Return available models to the UI."""
    models = [
        {"id": "gemma2",       "name": "Gemma 2 (Local · Offline)",    "provider": "ollama", "badge": "LOCAL"},
        {"id": "qwen2.5",      "name": "Qwen 2.5 (Local · Fast)",      "provider": "ollama", "badge": "LOCAL"},
        {"id": "gemini-cloud", "name": "Gemini 2.5 Flash (Cloud)",      "provider": "vertex", "badge": "CLOUD"},
    ]
    # Probe which Ollama models are actually running
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            available = {m["name"] for m in r.json().get("models", [])}
            for m in models:
                if m["provider"] == "ollama":
                    tag = OLLAMA_MODELS.get(m["id"], "")
                    m["available"] = tag in available
            ollama_ok = True
    except Exception:
        for m in models:
            if m["provider"] == "ollama":
                m["available"] = False

    if not ollama_ok:
        for m in models:
            if m["provider"] == "ollama":
                m["available"] = False

    for m in models:
        if m["provider"] == "vertex":
            m["available"] = vertex_client is not None

    return jsonify(models)


@app.route("/agent", methods=["POST"])
def agent():
    try:
        data          = request.json or {}
        command       = data.get("command", "")
        instruction   = data.get("instruction")
        context       = data.get("context", "")[:MAX_CONTEXT]
        prev_context  = data.get("previous_context", "")
        last_action   = data.get("last_action", "")
        repeat_count  = data.get("repeat_count", 0)
        model_key     = data.get("model", "gemma2")   # ← NEW: model chosen in UI

        log("INFO", f"Command: {command} | Model: {model_key} | Mode: {instruction}")

        task_type    = instruction if instruction else detect_task_type(command)
        mode_guidance = f"MODE: {instruction.upper()}" if instruction else "MODE: GENERAL"

        if task_type == "coding":
            prompt = (
                f"{CODING_PROMPT}\nTASK: {command}\n{mode_guidance}\n"
                f"PAGE: {context[:4000]}\nLAST ACTION: {last_action}"
            )
        else:
            prompt = (
                f"{SYSTEM_PROMPT}\nGOAL: {command}\n{mode_guidance}\n"
                f"PAGE: {context[:4000]}\nPREVIOUS: {prev_context[:2000]}\n"
                f"LAST ACTION: {last_action}"
            )

        # ── Call chosen LLM ──
        raw_text = generate_response(model_key, prompt)

        try:
            result = parse_llm_json(raw_text)
        except Exception as e:
            log("ERROR", f"Parsing Error: {e} | Raw: {raw_text[:200]}")
            result = {"action": "wait", "reason": "Parsing failed", "confidence": 0}

        # Enhance selector
        if not result.get("selector") and result.get("value") and result.get("action") in ["click", "type"]:
            result["fallback_selector"] = f"*:contains('{result['value']}')"

        # Navigation safety
        if result.get("action") in ["open_new_tab", "navigate"]:
            result["action"] = "open_new_tab"
            if any(k in command.lower() for k in ["open", "go to", "visit"]) and len(command.split()) < 4:
                result["is_goal_achieved"] = True

        # Coding state machine
        if task_type == "coding":
            code = result.get("value", "")
            if last_action != "type" and any(k in code for k in ["class ", "def ", "function", "import "]):
                result["action"]    = "type"
                result["selector"]  = "textarea, .CodeMirror, [contenteditable='true'], .monaco-editor"
                result["caption"]   = "Writing and pasting generated code..."
            elif last_action == "type":
                result["action"]    = "click"
                result["selector"]  = "button[type='submit'], .submit-btn, button:contains('Submit')"
                result["caption"]   = "Submitting your code solution..."
            elif last_action == "click":
                result["action"]          = "stop"
                result["is_goal_achieved"] = True
                result["caption"]         = "Coding task successfully completed!"

        # Save to history
        if history_col is not None and command:
            try:
                history_col.insert_one({
                    "command":    command,
                    "timestamp":  datetime.now(),
                    "task_type":  task_type,
                    "model":      model_key,
                    "status":     "success"
                })
            except Exception:
                pass

        # Low-confidence fallback
        if result.get("confidence", 1) < 0.2:
            result["action"]  = "wait"
            result["caption"] = "Thinking carefully... confidence is low."

        final_res = {
            "action":           result.get("action"),
            "selector":         result.get("selector", ""),
            "fallback_selector":result.get("fallback_selector", ""),
            "value":            result.get("value", ""),
            "caption":          result.get("caption", result.get("reason", "Working...")),
            "reason":           result.get("reason", ""),
            "confidence":       result.get("confidence", 0.5),
            "is_goal_achieved": result.get("is_goal_achieved", False),
            "meta": {
                "repeat_count": repeat_count,
                "task":         task_type,
                "model":        model_key,
                "time":         datetime.now().isoformat()
            }
        }

        log("ACTION", f"{final_res['action']} (Conf: {final_res['confidence']}) via {model_key}")
        return jsonify(final_res)

    except Exception as e:
        error_msg = str(e)
        log("CRITICAL", error_msg)
        return jsonify({
            "action":           "chat",
            "value":            f"System Error: {error_msg}",
            "caption":          "Brain Error: Backing off...",
            "is_goal_achieved": True
        })


@app.route("/api/logs", methods=["GET"])
def get_logs():
    return jsonify(LOG_QUEUE)


@app.route("/history", methods=["GET"])
def get_history():
    if history_col is None:
        return jsonify([])
    try:
        cursor = history_col.find().sort("timestamp", -1).limit(20)
        logs   = [
            {
                "command": doc.get("command"),
                "time":    doc.get("timestamp").strftime("%H:%M:%S"),
                "model":   doc.get("model", "unknown")
            }
            for doc in cursor
        ]
        return jsonify(logs)
    except Exception:
        return jsonify([])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)