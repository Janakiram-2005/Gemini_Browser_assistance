// ═══════════════════════════════════════════════════════════
//  AI Agent Pro — Side Panel JS
//  Supports: Gemma2 (local), Qwen2.5 (local), Gemini Cloud
// ═══════════════════════════════════════════════════════════

const API = "http://127.0.0.1:5000";

// ── ELEMENTS ───────────────────────────────────────────────
const taskInput  = document.getElementById("taskInput");
const modeSelect = document.getElementById("modeSelect");
const runBtn     = document.getElementById("runBtn");
const stopBtn    = document.getElementById("stopBtn");
const output     = document.getElementById("output");
const statBadge  = document.getElementById("statBadge");
const micBtn     = document.getElementById("micBtn");
const voiceTogBtn= document.getElementById("voiceTogBtn");
const histBtn    = document.getElementById("histBtn");
const histOverlay= document.getElementById("histOverlay");
const closeHist  = document.getElementById("closeHist");
const clearHistBtn= document.getElementById("clearHistBtn");
const histList   = document.getElementById("histList");
const modelPill  = document.getElementById("modelPill");
const pillTxt    = document.getElementById("pillTxt");

// ── STATE ──────────────────────────────────────────────────
let isRunning     = false;
let prevCtx       = "";
let lastAct       = "";
let repeatCount   = 0;
let voiceEnabled  = true;
let selectedModel = "gemma2";   // default: local Gemma2
let allModels     = [];

// ── MODEL META ─────────────────────────────────────────────
const MODEL_META = {
  "gemma2":       { icon: "🦙", name: "Gemma 2",    sub: "Google · 5.4 GB · Offline",  type: "local" },
  "qwen2.5":      { icon: "⚡", name: "Qwen 2.5",   sub: "Alibaba · 397 MB · Turbo",   type: "local" },
  "gemini-cloud": { icon: "✨", name: "Gemini 2.5", sub: "Google Cloud · Latest",       type: "cloud" },
};

// ═══════════════════════════════════════════════════════════
//  1. MODEL LOADING & RENDERING
// ═══════════════════════════════════════════════════════════
async function loadModels() {
  try {
    const r = await fetch(`${API}/api/models`);
    allModels = await r.json();
    renderModels();
  } catch (e) {
    // Backend offline — show stubs
    allModels = [
      { id: "gemma2",       name: "Gemma 2",    badge: "LOCAL", provider: "ollama", available: false },
      { id: "qwen2.5",      name: "Qwen 2.5",   badge: "LOCAL", provider: "ollama", available: false },
      { id: "gemini-cloud", name: "Gemini 2.5", badge: "CLOUD", provider: "vertex", available: false },
    ];
    renderModels();
    output.textContent = "⚠️ Backend offline. Start app.py first.";
  }
}

function renderModels() {
  const row = document.getElementById("modelRow");
  row.innerHTML = "";

  allModels.forEach(m => {
    const mt      = MODEL_META[m.id] || { icon: "🤖", name: m.name, sub: "", type: "local" };
    const isLocal = mt.type === "local";
    const avail   = m.available !== false;
    const isAct   = m.id === selectedModel;

    const card = document.createElement("div");
    card.className = [
      "mcard",
      isLocal ? "local" : "cloud",
      isAct   ? "active"   : "",
      !avail  ? "disabled" : ""
    ].filter(Boolean).join(" ");
    card.dataset.id = m.id;

    card.innerHTML = `
      <span class="mcard-icon">${mt.icon}</span>
      <div class="mcard-body">
        <div class="mcard-name">${mt.name}</div>
        <div class="mcard-sub">${mt.sub}</div>
      </div>
      <div class="mcard-right">
        <span class="mbadge ${isLocal ? "local" : "cloud"}">${m.badge || "LOCAL"}</span>
        <span class="sdot ${avail ? "on" : "off"}"></span>
      </div>
      <span class="check-ring">${isAct ? "✓" : ""}</span>
    `;

    card.addEventListener("click", () => {
      if (!avail) return;
      selectedModel = m.id;
      renderModels();
      savePref("selectedModel", selectedModel);
      updatePill();
    });

    row.appendChild(card);
  });

  updatePill();
}

function updatePill() {
  const m  = allModels.find(x => x.id === selectedModel) || {};
  const mt = MODEL_META[selectedModel] || {};
  modelPill.className = "model-pill" + (mt.type === "cloud" ? " cloud" : "");
  pillTxt.textContent = (mt.name || selectedModel) + (mt.type === "local" ? " · Offline" : " · Cloud");
}

// ═══════════════════════════════════════════════════════════
//  2. AGENT LOOP
// ═══════════════════════════════════════════════════════════
async function runAgent(command) {
  if (!command || isRunning) return;

  isRunning   = true;
  prevCtx     = "";
  lastAct     = "";
  repeatCount = 0;

  setUIBusy(true);
  output.textContent = "🚀 Initialising…";

  while (isRunning) {
    try {
      // A. CAPTURE PAGE
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab || !isRunning) break;

      let context = "";
      try {
        const res = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => document.body?.innerText?.slice(0, 8000) || ""
        });
        context = res[0]?.result || "";
      } catch (_) {}

      // B. BUILD PAYLOAD
      const payload = {
        command,
        model:            selectedModel,
        context,
        previous_context: prevCtx,
        last_action:      lastAct,
        repeat_count:     repeatCount,
      };
      const instr = modeSelect.value;
      if (instr) payload.instruction = instr;

      console.log(`📡 [Agent] model=${selectedModel} action=${lastAct} repeat=${repeatCount}`);

      // C. CALL BACKEND
      const resp = await fetch(`${API}/agent`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload)
      });

      if (resp.status === 429) {
        output.textContent = "⚠️ Rate limit — cooling down 10s…";
        await sleep(10000);
        continue;
      }
      if (!resp.ok) {
        const e = await resp.json().catch(() => ({}));
        throw new Error(e.value || `HTTP ${resp.status}`);
      }

      const data   = await resp.json();
      const action = (data?.action || "").trim().toLowerCase();

      // D. VALIDATE
      const VALID = ["open_new_tab","click","type","scroll","close_tab","switch_tab","wait","stop","navigate","chat"];
      if (!action || !VALID.includes(action)) {
        await sleep(2000); continue;
      }

      if (!isRunning) break;

      // E. LOOP GUARD
      if (action === lastAct) repeatCount++;
      else repeatCount = 0;

      // F. EXECUTE
      await executeAction(data);

      // G. STOP CONDITIONS
      if (action === "stop" || data.is_goal_achieved || repeatCount >= 3) {
        const msg = data.caption || data.value || "✅ Task complete!";
        output.textContent = msg;
        if (voiceEnabled) speak(msg);
        isRunning = false;
        break;
      }

      // H. ADVANCE STATE
      prevCtx = context;
      lastAct = action;
      await sleep(3000);

    } catch (err) {
      console.error("Agent error:", err);
      output.textContent = `❌ Error: ${err.message}`;
      isRunning = false;
      break;
    }
  }

  setUIBusy(false);
}

// ═══════════════════════════════════════════════════════════
//  3. ACTION EXECUTOR
// ═══════════════════════════════════════════════════════════
async function executeAction(data) {
  const action = (data?.action || "").trim().toLowerCase();
  const { selector, value, caption } = data;

  output.textContent = caption || `Executing: ${action}…`;
  statBadge.textContent = action.toUpperCase();

  if (value && (value.startsWith("chrome://") || value.startsWith("chrome-extension://"))) return;

  try {
    switch (action) {
      case "open_new_tab":
      case "navigate":
        await chrome.tabs.create({ url: value });
        break;

      case "click":
      case "type":
      case "scroll":
        await runScript(action, data);
        break;

      case "close_tab":
        const [t] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (t) await chrome.tabs.remove(t.id);
        break;

      case "switch_tab":
        window.focus();
        break;

      case "wait":
        await sleep(1500);
        break;

      case "chat":
        output.textContent = value || "Agent response received.";
        if (voiceEnabled) speak(output.textContent);
        break;
    }
  } catch (e) { console.error("Executor:", e) }
}

// ── DOM SCRIPT ─────────────────────────────────────────────
async function runScript(type, data) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;

  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    args:   [type, data],
    func: (action, params) => {
      function findEl() {
        let el = document.querySelector(params.selector);
        if (!el && params.value && ["click","type"].includes(action)) {
          const low = params.value.toLowerCase();
          el = [...document.querySelectorAll("button,a,div,span,input,textarea")]
                .find(e => (e.innerText || e.value || "").toLowerCase().includes(low));
        }
        if (!el && action === "type") el = document.querySelector("textarea,input,[contenteditable='true']");
        return el;
      }
      const el = findEl();
      if (!el && action !== "scroll") return;
      try {
        if (action === "click") { el.click(); return; }
        if (action === "scroll") { window.scrollBy(0, 500); return; }
        if (action === "type") {
          el.focus();
          if (el.classList?.contains("CodeMirror") && el.CodeMirror) {
            el.CodeMirror.setValue(""); el.CodeMirror.setValue(params.value);
          } else if (el.contentEditable === "true") {
            el.innerText = params.value;
          } else {
            el.value = ""; el.value = params.value;
          }
          el.dispatchEvent(new Event("input", { bubbles: true }));
        }
      } catch (_) {}
    }
  });
}

// ═══════════════════════════════════════════════════════════
//  4. VOICE
// ═══════════════════════════════════════════════════════════
let recognition = null;

function normalise(t) {
  return t.toLowerCase()
    .replace("lead code","leetcode")
    .replace("open mail","open gmail")
    .trim();
}

function speak(text) {
  if (!voiceEnabled) return;
  const synth = window.speechSynthesis;
  const u     = new SpeechSynthesisUtterance(text);
  const voices = synth.getVoices();
  u.voice = voices.find(v => v.name.includes("Google") && v.lang.startsWith("en")) || voices[0];
  synth.cancel();
  synth.speak(u);
}

micBtn.addEventListener("click", async () => {
  // unlock audio
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));

  if (recognition) { recognition.stop(); recognition = null; micBtn.classList.remove("rec"); return; }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { output.textContent = "⚠️ Speech not supported."; return; }

  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true });
    s.getTracks().forEach(t => t.stop());
  } catch (_) {}

  recognition = new SR();
  recognition.lang = "en-US";
  recognition.continuous = true;
  recognition.interimResults = true;

  micBtn.classList.add("rec");

  recognition.onresult = e => {
    const last = e.results[e.results.length - 1];
    const txt  = normalise(last[0].transcript);
    taskInput.value = txt;
    if (last.isFinal) {
      recognition.stop(); recognition = null; micBtn.classList.remove("rec");
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
      runAgent(txt);
    }
  };
  recognition.onend = () => { recognition = null; micBtn.classList.remove("rec"); };
  recognition.start();
});

// ═══════════════════════════════════════════════════════════
//  5. UI HELPERS
// ═══════════════════════════════════════════════════════════
function setUIBusy(busy) {
  runBtn.style.display  = busy ? "none"  : "flex";
  stopBtn.style.display = busy ? "block" : "none";
  statBadge.className   = busy ? "badge running" : "badge";
  statBadge.textContent = busy ? "RUNNING" : "IDLE";
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── PREFERENCES ────────────────────────────────────────────
function savePref(k, v) {
  try { chrome.storage.local.set({ [k]: v }); } catch (_) {}
}
async function loadPrefs() {
  try {
    const p = await chrome.storage.local.get(["selectedModel","voiceEnabled"]);
    if (p.selectedModel)  selectedModel = p.selectedModel;
    if (p.voiceEnabled !== undefined) {
      voiceEnabled = p.voiceEnabled;
      voiceTogBtn.className = "hbtn" + (voiceEnabled ? " act" : "");
      voiceTogBtn.innerHTML = `<i class="fas fa-headphones"></i> ${voiceEnabled ? "Voice" : "Muted"}`;
    }
  } catch (_) {}
}

// ═══════════════════════════════════════════════════════════
//  6. EVENT BINDINGS
// ═══════════════════════════════════════════════════════════
runBtn.addEventListener("click", () => {
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
  runAgent(taskInput.value.trim());
});

stopBtn.addEventListener("click", () => {
  isRunning = false;
  setUIBusy(false);
  output.textContent = "⏹ Stopped by user.";
});

taskInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
    runAgent(taskInput.value.trim());
  }
});

voiceTogBtn.addEventListener("click", () => {
  voiceEnabled = !voiceEnabled;
  savePref("voiceEnabled", voiceEnabled);
  voiceTogBtn.className = "hbtn" + (voiceEnabled ? " act" : "");
  voiceTogBtn.innerHTML = `<i class="fas fa-headphones"></i> ${voiceEnabled ? "Voice" : "Muted"}`;
  if (voiceEnabled) speak("Voice enabled");
});

// ── HISTORY ────────────────────────────────────────────────
histBtn.addEventListener("click", () => {
  histOverlay.style.display = "block";
  loadHistory();
});

closeHist.addEventListener("click", () => {
  histOverlay.style.display = "none";
});

clearHistBtn.addEventListener("click", () => {
  histList.innerHTML = '<div style="color:var(--tx3);font-size:.78rem;text-align:center;padding:20px">History cleared</div>';
});

async function loadHistory() {
  histList.innerHTML = '<div style="color:var(--tx3);font-size:.78rem;">Loading…</div>';
  try {
    const r    = await fetch(`${API}/history`);
    const data = await r.json();
    if (!data.length) {
      histList.innerHTML = '<div style="color:var(--tx3);font-size:.78rem;text-align:center;padding:20px">No history yet</div>';
      return;
    }
    histList.innerHTML = "";
    data.forEach(item => {
      const isCloud = item.model === "gemini-cloud";
      const d = document.createElement("div");
      d.className = "hist-item";
      d.innerHTML = `
        <div class="hist-cmd">${escHtml(item.command)}</div>
        <div class="hist-meta">
          <div>${item.time || ""}</div>
          <span class="hist-model-tag ${isCloud ? "cloud" : ""}">${item.model || "unknown"}</span>
        </div>
      `;
      d.addEventListener("click", () => {
        taskInput.value = item.command;
        histOverlay.style.display = "none";
      });
      histList.appendChild(d);
    });
  } catch (e) {
    histList.innerHTML = '<div style="color:var(--danger);font-size:.78rem;">Could not load history — is the server running?</div>';
  }
}

function escHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ═══════════════════════════════════════════════════════════
//  7. BOOT
// ═══════════════════════════════════════════════════════════
(async () => {
  await loadPrefs();
  await loadModels();
  // Refresh model availability every 30 s
  setInterval(loadModels, 30000);
})();