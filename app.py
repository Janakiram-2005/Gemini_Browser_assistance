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
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "30"))   # 30-second default active step timeout threshold

# ──────────────────────────────────────────────
# FRAME CACHE (dHash Layout Cache)
# ──────────────────────────────────────────────
FRAME_CACHE = {}  # dHash -> cached visual layout data / response

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
LOG_QUEUE      = []
IN_MEM_HISTORY = []   # fallback when MongoDB is unavailable
MAX_CONTEXT    = 6000
MAX_REPEAT     = 2

# ── LIVE TASK STATUS ──────────────────────────────────────────
LIVE_STATUS = {
    "running":        False,
    "task_id":        None,
    "command":        "",
    "step":           0,
    "current_action": "idle",
    "model":          "",
    "started_at":     None,
}

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
You are the planner and reasoning engine. You decide the best SINGLE next step.
The frontend executes each action you return, then calls you again with updated page context.

CRITICAL RULES:
1. Return EXACTLY ONE action per response.
2. Do NOT set is_goal_achieved=true until ALL parts of the goal are complete.
3. Do NOT engage in conversation — only return JSON.
4. Use `CURRENT PAGE URL` to determine your location. If you are already on the target domain, do NOT call "open_new_tab" again.
5. If the goal requires Gmail folders (e.g., spam, sent, trash, inbox), navigate directly by calling "open_new_tab" with the direct URL, which is extremely reliable:
   - Spam: https://mail.google.com/mail/u/0/#spam
   - Inbox: https://mail.google.com/mail/u/0/#inbox
   - Sent: https://mail.google.com/mail/u/0/#sent
   - Trash: https://mail.google.com/mail/u/0/#trash
6. "open_new_tab" just opens the URL — it does NOT complete multi-step goals.
7. Use the "VISIBLE LINKS & BUTTONS ON PAGE" list to match against elements you want to click or interact with. Prefer exact CSS selectors when available.
8. ALWAYS check the "ACTIONS PREVIOUSLY COMPLETED IN THIS TASK" section to see what actions you have already executed. Do NOT repeat or navigate back to steps that are already completed. Progress forward to the next uncompleted part of the goal.
9. Task Division: You must divide the main goal into 2 to 5 explicit sequential subtasks in Step 1, returning them as a list of subtasks in the `subtasks` array. In subsequent turns, you will receive the previous `subtasks` array, update the status of each subtask (choose between `pending`, `active`, or `completed`) based on your current step progress, and return the updated `subtasks` array. Make sure you only have ONE `active` subtask at a time.

MULTI-STEP EXAMPLE (Goal: "open youtube and search for Believer"):
  Step 1: {"action":"open_new_tab",  "value":"https://www.youtube.com", "is_goal_achieved":false, "subtasks":[{"title":"Open YouTube","status":"active"},{"title":"Search for Believer","status":"pending"}]}
  (Next turn context shows CURRENT PAGE URL: https://www.youtube.com/)
  Step 2: {"action":"click",  "selector":"input#search", "value":"search box", "is_goal_achieved":false, "subtasks":[{"title":"Open YouTube","status":"completed"},{"title":"Search for Believer","status":"active"}]}
  Step 3: {"action":"type",   "selector":"input#search", "value":"Believer", "is_goal_achieved":false, "subtasks":[{"title":"Open YouTube","status":"completed"},{"title":"Search for Believer","status":"active"}]}
  Step 4: {"action":"click",  "selector":"button#search-icon-legacy", "value":"search button", "is_goal_achieved":false, "subtasks":[{"title":"Open YouTube","status":"completed"},{"title":"Search for Believer","status":"active"}]}
  Step 5: {"action":"chat",   "value":"Done! Searched YouTube for Believer.", "is_goal_achieved":true, "subtasks":[{"title":"Open YouTube","status":"completed"},{"title":"Search for Believer","status":"completed"}]}

AVAILABLE ACTIONS:
- open_new_tab: value="FULL URL including https://"
- close_tab: (no value)
- switch_tab: (no value)
- click: selector="CSS selector", value="Visible text fallback"
- type: selector="CSS selector", value="Text to input"
- key: value="Enter" (press a key after typing)
- scroll: value="up | down"
- wait: (no value, use when page is loading)
- stop: (no value)
- chat: value="Message to user"

COMMON SELECTORS:
- YouTube search input: input#search
- YouTube search button: button#search-icon-legacy
- Google search input: textarea[name='q']
- Google search button: input[name='btnK']
- Gmail compose: div[role='button'][gh='cm']

OUTPUT FORMAT — reply with ONLY this JSON, no other text:
{
  "action": "<action_name>",
  "selector": "<CSS selector or empty string>",
  "value": "<URL, text to type, or message>",
  "caption": "<User-friendly status message>",
  "reason": "<Why this step?>",
  "confidence": <0.0 to 1.0>,
  "is_goal_achieved": <true only when ALL goal steps are done>,
  "subtasks": [
    {"title": "Subtask title description", "status": "pending | active | completed"}
  ]
}
"""

EXECUTE_MODE_PROMPT = """
⚡ EXECUTE MODE ACTIVATED:
Your objective is decisive, rapid interaction.
- Prioritize click, type, navigate, and key actions.
- Do NOT spend time explaining or summarizing.
- Complete form submissions, button interactions, and navigation flows as quickly as possible.
"""

READ_MODE_PROMPT = """
📖 READ MODE ACTIVATED:
Your objective is reading comprehension, text extraction, and summarization.
- Prioritize scrolling, waiting, and analyzing the page content.
- Do NOT click on random links or compose text unless strictly required to access the reading material.
- Gather all text context. Once gathered, use the "chat" action to summarize or answer the user's question directly.
"""

SCAN_MODE_PROMPT = """
🔍 SCAN MODE ACTIVATED:
Your objective is scanning, indexing, and finding specific elements, links, prices, or keywords.
- Prioritize scrolling and extracting lists of matched elements, products, prices, or links.
- Do NOT click buy buttons, delete buttons, or modify application state.
- Gather a clean list of findings and present them to the user via the "chat" action once completed.
"""

WRITE_MODE_PROMPT = """
✏️ WRITE MODE ACTIVATED:
Your objective is detailed text composition, email drafting, or form input writing.
- Focus on finding compose fields, textareas, inputs, and contenteditable elements.
- Compose detailed, professional, and contextually rich text.
- Execute "type" actions with the composed text, and use "key" (Enter) to submit or save when ready.
"""

SEARCH_EXECUTE_PROMPT = """
🌐 SEARCH & EXECUTE MODE ACTIVATED:
Your objective is to find information via search engines first, then execute a follow-up task.
- Step 1: Open a search engine (Google, Bing, or YouTube).
- Step 2: Search for the user query using the search bar.
- Step 3: Analyze the search result links from the DOM and click the most relevant match.
- Step 4: Perform the final task, compose the answer, or execute follow-up steps on the target page.
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
    """
    Call a local Ollama model using streaming mode.
    Streaming avoids a single long wait — tokens arrive as they're generated.
    Falls back gracefully on timeout.
    """
    model_name = OLLAMA_MODELS.get(model_key, model_key)
    log("LLM", f"Using Ollama → {model_name} (timeout={OLLAMA_TIMEOUT}s)")

    # ── Trim prompt for slow/large models to reduce token count ──
    # Keep system prompt + goal, but limit page context to 2000 chars
    if len(prompt) > 5000:
        # Find PAGE: marker and trim from there
        if "\nPAGE:" in prompt:
            parts = prompt.split("\nPAGE:", 1)
            page_section = parts[1][:2000]
            prompt = parts[0] + "\nPAGE:" + page_section

    payload = {
        "model":  model_name,
        "prompt": prompt,
        "stream": True,       # ← streaming: tokens arrive incrementally, no single huge wait
        "options": {
            "temperature": 0,
            "num_predict": 300,   # cap output tokens — we only need a short JSON action
        }
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=OLLAMA_TIMEOUT
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"Ollama '{model_name}' timed out after {OLLAMA_TIMEOUT}s. "
            f"Try: (1) use qwen2.5 (faster) or (2) set OLLAMA_TIMEOUT=300 in .env"
        )
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Run: ollama serve"
        )

    # Collect streamed tokens
    full_text = []
    try:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            chunk = json.loads(raw_line.decode("utf-8"))
            token = chunk.get("response", "")
            full_text.append(token)
            if chunk.get("done"):
                break
    except Exception:
        pass  # stream closed early — still try to parse what we got

    return "".join(full_text)


def call_ollama_with_fallback(model_key: str, prompt: str) -> str:
    """
    Try the requested Ollama model, fall back to qwen2.5 (fast, 397 MB),
    then to Vertex AI cloud as last resort.
    """
    # 1. Try the requested model
    try:
        return call_ollama(model_key, prompt)
    except (TimeoutError, ConnectionError) as e:
        log("WARN", f"Primary model '{model_key}' failed: {e}")
    except Exception as e:
        log("WARN", f"Primary Ollama error: {e}")

    # 2. Fallback to qwen2.5 (small, fast)
    if model_key != "qwen2.5":
        try:
            log("LLM", "↪ Falling back to qwen2.5 (fast model)…")
            return call_ollama("qwen2.5", prompt)
        except Exception as e:
            log("WARN", f"qwen2.5 fallback failed: {e}")

    # 3. Fallback to Vertex AI cloud
    if vertex_client is not None:
        try:
            log("LLM", "↪ Falling back to Cloud Gemini (Vertex AI)…")
            return call_vertex(prompt)
        except Exception as e:
            log("WARN", f"Vertex fallback failed: {e}")

    raise RuntimeError(
        "All LLM backends failed. Check: "
        "(1) Is Ollama running? → run 'ollama serve'. "
        "(2) Is the model pulled? → run 'ollama pull gemma2'. "
        "(3) Try selecting qwen2.5 (faster) in the UI."
    )


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
    """Route the prompt to the correct LLM backend with automatic fallback."""
    if model_key == "gemini-cloud":
        # Cloud path — no Ollama needed
        try:
            return call_vertex(prompt)
        except Exception as e:
            log("WARN", f"Vertex AI failed: {e}. Trying Ollama fallback.")
            return call_ollama_with_fallback("qwen2.5", prompt)
    else:
        # Ollama path with fallback chain
        return call_ollama_with_fallback(model_key, prompt)



# ──────────────────────────────────────────────
# PARSE LLM OUTPUT → JSON
# ──────────────────────────────────────────────
def repair_truncated_json(text: str) -> str:
    """Attempt to repair simple truncated JSON by closing open quotes, brackets, and braces."""
    text = text.strip()
    idx = text.find("{")
    if idx != -1:
        text = text[idx:]
    else:
        return text
        
    in_quote = False
    escape = False
    stack = []
    repaired_chars = []
    
    for c in text:
        if escape:
            repaired_chars.append(c)
            escape = False
            continue
        if c == '\\':
            repaired_chars.append(c)
            escape = True
            continue
        if c == '"':
            in_quote = not in_quote
            repaired_chars.append(c)
            continue
            
        if not in_quote:
            if c in '{[':
                stack.append(c)
            elif c in '}]':
                if stack:
                    stack.pop()
        repaired_chars.append(c)
        
    if in_quote:
        repaired_chars.append('"')
        
    while stack:
        opened = stack.pop()
        if opened == '{':
            repaired_chars.append('}')
        elif opened == '[':
            repaired_chars.append(']')
            
    return "".join(repaired_chars)


def parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM output, handling Ollama/Llama quirks like extra text."""
    text = text.strip()

    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first JSON object from text (handles Llama adding explanation text)
    import re
    match = re.search(r'\{[\s\S]*?\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to repair and parse the largest JSON-like block
    repaired = repair_truncated_json(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Last resort: find largest JSON block from original
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in LLM output: {text[:300]}")

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
        import uuid, time as _time
        data          = request.json or {}
        command       = data.get("command", "")
        instruction   = data.get("instruction")
        context       = data.get("context", "")[:MAX_CONTEXT]
        prev_context  = data.get("previous_context", "")
        last_action   = data.get("last_action", "")
        repeat_count  = data.get("repeat_count", 0)
        step_number   = data.get("step_number", 1)
        model_key     = data.get("model", "gemma2")
        page_url      = data.get("page_url", "")
        page_links    = data.get("page_links", [])
        action_history = data.get("action_history", [])
        subtasks      = data.get("subtasks", [])

        # Visual Fallback & Differential Frame Cache parameters
        frame_hash           = data.get("frame_hash", "")
        frame_data           = data.get("frame_data", "")
        use_visual_fallback  = data.get("use_visual_fallback", False)
        dhash_diff           = data.get("dhash_diff", 0)
        frame_size_kb        = data.get("frame_size_kb", 0)
        last_confidence      = data.get("last_confidence", 1.0)
        is_canvas_interaction= data.get("is_canvas_interaction", False)
        frame_cached         = data.get("frame_cached", False)

        # Update live status
        LIVE_STATUS["running"]        = True
        LIVE_STATUS["command"]        = command
        LIVE_STATUS["step"]           = step_number
        LIVE_STATUS["current_action"] = last_action or "starting"
        LIVE_STATUS["model"]          = model_key
        if not LIVE_STATUS.get("started_at") or last_action == "":
            LIVE_STATUS["task_id"]    = str(uuid.uuid4())[:8]
            LIVE_STATUS["started_at"] = datetime.now().isoformat()

        log("INFO", f"[Step {step_number}] Command: {command} | Model: {model_key} | Mode: {instruction}")

        # Check visual fallback trigger conditions
        visual_info = ""
        is_visual_triggered = use_visual_fallback or is_canvas_interaction or (last_confidence < 0.70 and step_number > 1)
        if is_visual_triggered:
            trigger_reasons = []
            if last_confidence < 0.70 and step_number > 1:
                trigger_reasons.append(f"DOM confidence low ({last_confidence:.2f} < 0.70)")
            if is_canvas_interaction:
                trigger_reasons.append("Canvas element interaction")
            if use_visual_fallback and not trigger_reasons:
                trigger_reasons.append("Visual fallback requested")

            trigger_str = ", ".join(trigger_reasons)
            log("VISUAL", f"[Step {step_number}] Visual Fallback Triggered ({trigger_str}) | dHash: {frame_hash or 'N/A'}")

            if frame_cached or (frame_hash and frame_hash in FRAME_CACHE):
                log("CACHE", f"[Step {step_number}] dHash Frame Cache HIT [{frame_hash}] - Reusing cached visual layout data")
                visual_info = f"\n[VISUAL FRAME CACHE HIT] dHash: {frame_hash} (Static Screen State - Visual VLM Calls Bypassed)\n"
            elif frame_data or frame_hash:
                if frame_size_kb > 300:
                    log("WARN", f"[Step {step_number}] Visual frame payload {frame_size_kb:.1f} KB exceeds 300 KB constraint")
                else:
                    log("VISUAL", f"[Step {step_number}] Visual frame payload received: {frame_size_kb:.1f} KB (<300 KB limit)")
                visual_info = f"\n[VISUAL FALLBACK FRAME ATTACHED] dHash: {frame_hash} | Payload Size: {frame_size_kb:.1f} KB\n"
                if frame_hash:
                    FRAME_CACHE[frame_hash] = {
                        "timestamp": datetime.now().isoformat(),
                        "url": page_url,
                        "size_kb": frame_size_kb
                    }
        else:
            log("DOM", f"[Step {step_number}] DOM-First Execution (Visual VLM calls bypassed)")

        task_type    = instruction if instruction else detect_task_type(command)
        mode_guidance = f"MODE: {instruction.upper()}" if instruction else "MODE: GENERAL"

        # Format links for the prompt context
        links_str = ""
        if page_links:
            links_str = "VISIBLE LINKS & BUTTONS ON PAGE:\n"
            for link in page_links:
                text = link.get("text", "").strip()
                sel = link.get("sel", "").strip()
                if text:
                    links_str += f"- Text: \"{text}\" | Selector: \"{sel}\"\n"

        # Format action history for episodic memory
        history_str = ""
        if action_history:
            history_str = "ACTIONS PREVIOUSLY COMPLETED IN THIS TASK:\n"
            for item in action_history:
                step = item.get("step", 1)
                act = item.get("action", "")
                val = item.get("value", "")
                cap = item.get("caption", "")
                history_str += f"- Step {step}: {act} (Value: \"{val}\", Caption: \"{cap}\")\n"

        # Format subtasks roadmap status for task division
        subtasks_str = ""
        if subtasks:
            subtasks_str = "CURRENT TASK DIVISION & ROADMAP STATUS:\n"
            for item in subtasks:
                title = item.get("title", "")
                status = item.get("status", "pending")
                subtasks_str += f"- Subtask: \"{title}\" (Status: {status})\n"

        page_info = f"CURRENT PAGE URL: {page_url}\n" if page_url else ""
        if visual_info:
            page_info += f"{visual_info}\n"
        if subtasks_str:
            page_info += f"\n{subtasks_str}\n"
        if history_str:
            page_info += f"\n{history_str}\n"
        if links_str:
            page_info += f"\n{links_str}\n"

        # Select mode specific prompts
        mode_prompt = ""
        if task_type == "execute":
            mode_prompt = EXECUTE_MODE_PROMPT
        elif task_type == "read":
            mode_prompt = READ_MODE_PROMPT
        elif task_type == "scan":
            mode_prompt = SCAN_MODE_PROMPT
        elif task_type == "write":
            mode_prompt = WRITE_MODE_PROMPT
        elif task_type == "search_execute":
            mode_prompt = SEARCH_EXECUTE_PROMPT

        if task_type == "coding":
            prompt = (
                f"{CODING_PROMPT}\nTASK: {command}\n{mode_guidance}\n"
                f"{page_info}"
                f"PAGE: {context[:4000]}\nLAST ACTION: {last_action}"
            )
        else:
            combined_system = SYSTEM_PROMPT
            if mode_prompt:
                combined_system = f"{SYSTEM_PROMPT}\n\n{mode_prompt}"
                
            prompt = (
                f"{combined_system}\nGOAL: {command}\n{mode_guidance}\n"
                f"{page_info}"
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

        # Navigation safety — normalise action name, never mark goal done here
        # (the LLM must decide when ALL steps are complete via is_goal_achieved)
        if result.get("action") in ["open_new_tab", "navigate"]:
            result["action"] = "open_new_tab"
            
            # Guard: Prevent redundant open_new_tab calls to the same page URL
            target_url = result.get("value", "").lower().strip()
            if target_url and page_url:
                from urllib.parse import urlparse
                try:
                    target_parsed = urlparse(target_url)
                    current_parsed = urlparse(page_url.lower())
                    target_domain = target_parsed.netloc
                    current_domain = current_parsed.netloc
                    
                    if target_domain and current_domain and target_domain == current_domain:
                        target_path_hash = target_parsed.path.rstrip('/') + ('#' + target_parsed.fragment if target_parsed.fragment else '')
                        current_path_hash = current_parsed.path.rstrip('/') + ('#' + current_parsed.fragment if current_parsed.fragment else '')
                        if target_path_hash == current_path_hash:
                            log("WARN", f"Redundant open_new_tab to active page detected: {target_url} — converting to wait")
                            result["action"] = "wait"
                            result["caption"] = "Already on target page. Preparing next actions..."
                except Exception as ex:
                    log("WARN", f"Redundant url parsing failed: {ex}")

            # Only mark done if the command is PURELY a navigation request
            # (single step: e.g. "open youtube.com") with no compound task
            compound_keywords = ["and", "then", "search", "find", "type", "click",
                                  "write", "send", "submit", "fill", "login", "sign"]
            is_simple_nav = (
                any(k in command.lower() for k in ["open", "go to", "visit", "navigate"])
                and len(command.split()) <= 4
                and not any(ck in command.lower() for ck in compound_keywords)
            )
            if is_simple_nav:
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

        # Save to history (first step only to avoid duplicate entries per task)
        if command and repeat_count == 0:
            hist_entry = {
                "id":        LIVE_STATUS.get("task_id", str(step_number)),
                "command":   command,
                "time":      datetime.now().strftime("%H:%M:%S"),
                "task_type": task_type,
                "model":     model_key,
                "status":    "success"
            }
            IN_MEM_HISTORY.append(hist_entry)
            if len(IN_MEM_HISTORY) > 100:
                IN_MEM_HISTORY.pop(0)
            if history_col is not None:
                try:
                    history_col.insert_one({
                        "command":   command,
                        "timestamp": datetime.now(),
                        "task_type": task_type,
                        "model":     model_key,
                        "status":    "success"
                    })
                except Exception:
                    pass

        # Low-confidence fallback
        if result.get("confidence", 1) < 0.2:
            result["action"]  = "wait"
            result["caption"] = "Thinking carefully... confidence is low."

        # Safety: cap repeat loops in backend too
        if repeat_count >= 5 and result.get("action") == last_action:
            log("WARN", f"Repeated action '{last_action}' x{repeat_count} — forcing stop")
            result["action"]           = "chat"
            result["value"]            = f"I seem to be stuck repeating '{last_action}'. Please try rephrasing your task."
            result["is_goal_achieved"] = True

        final_res = {
            "action":           result.get("action"),
            "selector":         result.get("selector", ""),
            "fallback_selector":result.get("fallback_selector", ""),
            "value":            result.get("value", ""),
            "caption":          result.get("caption", result.get("reason", "Working...")),
            "reason":           result.get("reason", ""),
            "confidence":       result.get("confidence", 0.5),
            "is_goal_achieved": result.get("is_goal_achieved", False),
            "step_number":      step_number,
            "subtasks":         result.get("subtasks", subtasks),
            "meta": {
                "repeat_count": repeat_count,
                "task":         task_type,
                "model":        model_key,
                "dhash":        frame_hash,
                "use_visual_fallback": is_visual_triggered,
                "frame_cached": frame_cached or (frame_hash in FRAME_CACHE),
                "frame_size_kb": frame_size_kb,
                "time":         datetime.now().isoformat()
            }
        }

        # Update live status with the chosen action
        LIVE_STATUS["current_action"] = final_res["action"]
        if final_res["is_goal_achieved"] or final_res["action"] in ["stop", "chat"]:
            LIVE_STATUS["running"] = False
            LIVE_STATUS["current_action"] = "idle"

        log("ACTION", f"{final_res['action']} (Conf: {final_res['confidence']}) via {model_key}")
        return jsonify(final_res)

    except Exception as e:
        error_msg = str(e)
        log("CRITICAL", error_msg)
        LIVE_STATUS["running"]        = False
        LIVE_STATUS["current_action"] = "error"
        return jsonify({
            "action":           "chat",
            "value":            f"System Error: {error_msg}",
            "caption":          "Brain Error: Backing off...",
            "is_goal_achieved": True,
            "meta": {
                "dhash":        frame_hash,
                "use_visual_fallback": is_visual_triggered,
                "frame_cached": frame_cached or (frame_hash in FRAME_CACHE),
                "frame_size_kb": frame_size_kb,
                "time":         datetime.now().isoformat()
            }
        })


@app.route("/api/logs", methods=["GET"])
def get_logs():
    return jsonify(LOG_QUEUE)


@app.route("/api/status", methods=["GET"])
def get_status():
    """Return live task execution state for frontends."""
    return jsonify(LIVE_STATUS)


@app.route("/history", methods=["GET"])
def get_history():
    # Try MongoDB first
    if history_col is not None:
        try:
            cursor = history_col.find().sort("timestamp", -1).limit(50)
            logs   = [
                {
                    "id":      str(doc.get("_id", "")),
                    "command": doc.get("command"),
                    "time":    doc.get("timestamp").strftime("%H:%M:%S"),
                    "model":   doc.get("model", "unknown")
                }
                for doc in cursor
            ]
            return jsonify(logs)
        except Exception:
            pass
    # Fallback to in-memory history
    return jsonify(list(reversed(IN_MEM_HISTORY[-50:])))


@app.route("/api/history/clear", methods=["DELETE"])
def clear_history():
    """Delete all history entries."""
    global IN_MEM_HISTORY
    IN_MEM_HISTORY = []
    if history_col is not None:
        try:
            history_col.delete_many({})
        except Exception:
            pass
    log("INFO", "History cleared by user")
    return jsonify({"ok": True, "message": "History cleared"})


@app.route("/api/history/<item_id>", methods=["DELETE"])
def delete_history_item(item_id):
    """Delete a single history entry by ID."""
    global IN_MEM_HISTORY
    # In-memory: ids are string indices
    IN_MEM_HISTORY = [h for h in IN_MEM_HISTORY if h.get("id") != item_id]
    if history_col is not None:
        try:
            from bson import ObjectId
            history_col.delete_one({"_id": ObjectId(item_id)})
        except Exception:
            pass
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)