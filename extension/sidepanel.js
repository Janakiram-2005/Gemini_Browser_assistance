// ═══════════════════════════════════════════════════════════
//  AI Agent Pro — Side Panel JS  (v2 — Full Feature Build)
//  Features: agentic loop, live console, voice in/out,
//            step breadcrumbs, live status, history w/ delete
// ═══════════════════════════════════════════════════════════

const API = "http://127.0.0.1:5000";

// ── ELEMENTS ───────────────────────────────────────────────
const taskInput   = document.getElementById("taskInput");
const modeSelect  = document.getElementById("modeSelect");
const runBtn      = document.getElementById("runBtn");
const stopBtn     = document.getElementById("stopBtn");
const output      = document.getElementById("output");
const statBadge   = document.getElementById("statBadge");
const statusBar   = document.getElementById("statusBar");
const statusIcon  = document.getElementById("statusIcon");
const statusText  = document.getElementById("statusText");
const stepCounter = document.getElementById("stepCounter");
const stepTrail   = document.getElementById("stepTrail");
const micBtn      = document.getElementById("micBtn");
const voiceTogBtn = document.getElementById("voiceTogBtn");
const histBtn     = document.getElementById("histBtn");
const histOverlay = document.getElementById("histOverlay");
const closeHist   = document.getElementById("closeHist");
const clearHistBtn= document.getElementById("clearHistBtn");
const histList    = document.getElementById("histList");
const modelPill   = document.getElementById("modelPill");
const pillTxt     = document.getElementById("pillTxt");
const consoleLines= document.getElementById("consoleLines");
const subtaskPanel = document.getElementById("subtaskPanel");
const subtaskList  = document.getElementById("subtaskList");

// ── STATE ──────────────────────────────────────────────────
let isRunning     = false;
let prevCtx       = "";
let lastAct       = "";
let repeatCount   = 0;
let stepNumber    = 0;
let voiceEnabled  = true;
let selectedModel = "gemma2";
let allModels     = [];
let stepHistory   = [];    // [{action, caption, done}]
let currentSubtasks = [];  // subtasks division roadmap
let logPollTimer  = null;
let lastLogCount  = 0;

// ── MODEL META ─────────────────────────────────────────────
const MODEL_META = {
  "gemma2":       { icon: "🦙", name: "Gemma 2",    sub: "Google · 5.4 GB · Offline",  type: "local" },
  "qwen2.5":      { icon: "⚡", name: "Qwen 2.5",   sub: "Alibaba · 397 MB · Turbo",   type: "local" },
  "gemini-cloud": { icon: "✨", name: "Gemini 2.5", sub: "Google Cloud · Latest",       type: "cloud" },
};

// ── ACTION META (for step chips) ───────────────────────────
const ACTION_ICONS = {
  open_new_tab: "🌐", navigate: "🌐", click: "👆", type: "⌨️",
  key: "↵", scroll: "📜", wait: "⏳", chat: "💬", stop: "🛑",
  coding: "💻", search: "🔍", close_tab: "❌", switch_tab: "🔄"
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
    allModels = [
      { id: "gemma2",       name: "Gemma 2",    badge: "LOCAL", provider: "ollama", available: false },
      { id: "qwen2.5",      name: "Qwen 2.5",   badge: "LOCAL", provider: "ollama", available: false },
      { id: "gemini-cloud", name: "Gemini 2.5", badge: "CLOUD", provider: "vertex", available: false },
    ];
    renderModels();
    addConsoleLine("WARN", "Backend offline — start app.py first");
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
      "mcard", isLocal ? "local" : "cloud",
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
  const mt = MODEL_META[selectedModel] || {};
  modelPill.className = "model-pill" + (mt.type === "cloud" ? " cloud" : "");
  pillTxt.textContent = (mt.name || selectedModel) + (mt.type === "local" ? " · Offline" : " · Cloud");
}

// ═══════════════════════════════════════════════════════════
//  2. LIVE STATUS HELPERS
// ═══════════════════════════════════════════════════════════
function setUIBusy(busy) {
  runBtn.disabled       = busy;
  runBtn.style.display  = busy ? "none"  : "flex";
  stopBtn.style.display = busy ? "block" : "none";
  statBadge.className   = busy ? "badge running" : "badge";
  statBadge.textContent = busy ? "RUNNING" : "IDLE";
  statusBar.className   = busy ? "status-bar running" : "status-bar";
  if (!busy) {
    statusIcon.textContent = "🟢";
    stepCounter.textContent = "";
  }
}

function updateStatus(icon, text, step) {
  statusIcon.textContent = icon;
  statusText.textContent = text;
  if (step !== undefined) stepCounter.textContent = `Step ${step}`;
}

// Action → human label
function actionLabel(action, value) {
  const labels = {
    open_new_tab: `Open ${(value||"").replace("https://","").split("/")[0]}`,
    navigate:     `Navigate`,
    click:        `Click element`,
    type:         `Type "${(value||"").slice(0,20)}${(value||"").length>20?"…":""}"`,
    key:          `Press ${value||"Enter"}`,
    scroll:       `Scroll ${value||"down"}`,
    wait:         `Wait…`,
    chat:         `Done`,
    stop:         `Complete`,
    close_tab:    `Close tab`,
    switch_tab:   `Switch tab`,
  };
  return labels[action] || action;
}

// ── STEP TRAIL ──────────────────────────────────────────────
function addStepChip(action, caption, isActive) {
  const chip = document.createElement("div");
  chip.className = "step-chip" + (isActive ? " active" : "");
  chip.innerHTML = `${ACTION_ICONS[action] || "▶"} ${caption || actionLabel(action)}`;
  chip.dataset.action = action;
  stepTrail.appendChild(chip);
  return chip;
}

function markLastChipDone() {
  const chips = stepTrail.querySelectorAll(".step-chip.active");
  chips.forEach(c => { c.classList.remove("active"); c.classList.add("done"); });
}

// ── SUBTASKS ROADMAP RENDERER ───────────────────────────────
function renderSubtasks() {
  if (!currentSubtasks || !currentSubtasks.length) {
    subtaskPanel.style.display = "none";
    return;
  }
  
  subtaskPanel.style.display = "block";
  subtaskList.innerHTML = "";
  
  currentSubtasks.forEach((st, idx) => {
    const item = document.createElement("div");
    const status = (st.status || "pending").toLowerCase();
    item.className = `subtask-item ${status}`;
    
    let icon = '<i class="far fa-circle" style="color:var(--tx3)"></i>';
    if (status === "completed") {
      icon = '<i class="fas fa-circle-check" style="color:var(--ok)"></i>';
    } else if (status === "active") {
      icon = '<i class="fas fa-spinner spin" style="color:var(--p)"></i>';
    }
    
    item.innerHTML = `
      <span class="subtask-icon">${icon}</span>
      <span class="subtask-title">${escHtml(st.title || `Subtask ${idx + 1}`)}</span>
      <span class="subtask-badge ${status}">${status}</span>
    `;
    subtaskList.appendChild(item);
  });
}

// ═══════════════════════════════════════════════════════════
//  3. AGENT LOOP
// ═══════════════════════════════════════════════════════════
async function runAgent(command) {
  if (!command || isRunning) return;

  isRunning   = true;
  prevCtx     = "";
  lastAct     = "";
  repeatCount = 0;
  stepNumber  = 0;
  stepHistory = [];
  currentSubtasks = [];
  renderSubtasks();
  stepTrail.innerHTML = "";

  setUIBusy(true);
  output.textContent = "🚀 Initialising task…";
  updateStatus("⚡", "Starting agent…");
  addConsoleLine("INFO", `New task: "${command.slice(0,60)}"`);

  while (isRunning) {
    try {
      stepNumber++;

      // A. CAPTURE PAGE — text + URL + visible links
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab || !isRunning) break;

      let context  = "";
      let pageUrl  = tab.url || "";
      let pageLinks = [];
      try {
        const res = await chrome.scripting.executeScript({
          target: { tabId: tab.id, allFrames: true },
          func: () => {
            function getDeepInnerText(root) {
              if (!root) return "";
              if (root.nodeType === 3) return root.textContent;
              if (root.nodeType === 1) {
                const style = window.getComputedStyle(root);
                if (style.display === 'none' || style.visibility === 'hidden') return "";
              }
              let text = "";
              for (const child of root.childNodes) {
                text += getDeepInnerText(child);
              }
              if (root.shadowRoot) {
                text += " " + getDeepInnerText(root.shadowRoot);
              }
              return text;
            }

            function getDeepElements(selector, root = document) {
              let list = [...root.querySelectorAll(selector)];
              const elements = root.querySelectorAll('*');
              for (const el of elements) {
                if (el.shadowRoot) {
                  list = list.concat(getDeepElements(selector, el.shadowRoot));
                }
              }
              return list;
            }

            const text  = getDeepInnerText(document.body).slice(0, 3000);
            const selStr = "a[href], [role='link'], [data-tooltip], button, input, textarea, label, [role='checkbox'], [role='radio']";
            const links = getDeepElements(selStr)
              .slice(0, 40)
              .map(el => {
                const inner = (el.innerText || el.value || el.getAttribute("data-tooltip") || el.title || "").trim().slice(0, 60);
                return {
                  text:    inner,
                  href:    el.href || el.getAttribute("href") || "",
                  tooltip: el.getAttribute("data-tooltip") || "",
                  sel:     el.getAttribute("data-tooltip")
                             ? `[data-tooltip="${el.getAttribute("data-tooltip")}"]`
                             : (el.id ? `#${el.id}` : (el.className?.split(" ")[0] ? `.${el.className.split(" ")[0]}` : el.tagName.toLowerCase()))
                };
              })
              .filter(l => l.text || l.tooltip);
            return { text, links, url: window.location.href };
          }
        });
        let mergedText = "";
        let mergedLinks = [];
        res.forEach(frame => {
          const r = frame.result || {};
          if (r.text) mergedText += "\n--- Frame: " + (r.url || "subframe") + " ---\n" + r.text + "\n";
          if (r.links) mergedLinks = mergedLinks.concat(r.links);
        });

        context   = mergedText.slice(0, 6000);
        pageLinks = mergedLinks;
      } catch (err) {
        console.error("Frame capture error:", err);
      }

      // B. BUILD PAYLOAD
      const payload = {
        command,
        model:            selectedModel,
        context,
        page_url:         pageUrl,
        page_links:       pageLinks.slice(0, 30),   // top 30 links with selectors
        previous_context: prevCtx,
        last_action:      lastAct,
        repeat_count:     repeatCount,
        step_number:      stepNumber,
        action_history:   stepHistory,
        subtasks:         currentSubtasks,
      };
      const instr = modeSelect.value;
      if (instr) payload.instruction = instr;

      updateStatus("🧠", `Thinking… (Step ${stepNumber})`, stepNumber);
      addConsoleLine("INFO", `[Step ${stepNumber}] Calling brain | last_action="${lastAct}"`);

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
      if (data && data.subtasks) {
        currentSubtasks = data.subtasks;
        renderSubtasks();
      }
      const action = (data?.action || "").trim().toLowerCase();

      // D. VALIDATE
      const VALID = ["open_new_tab","click","type","key","scroll","close_tab","switch_tab","wait","stop","navigate","chat","coding"];
      if (!action || !VALID.includes(action)) {
        addConsoleLine("WARN", `Unknown action: "${action}" — skipping`);
        await sleep(2000); continue;
      }

      if (!isRunning) break;

      // E. LOOP GUARD
      if (action === lastAct) repeatCount++;
      else repeatCount = 0;

      // F. UI UPDATE
      markLastChipDone();
      const chip = addStepChip(action, data.caption, true);
      updateStatus(ACTION_ICONS[action] || "▶", data.caption || actionLabel(action, data.value), stepNumber);
      output.textContent = data.caption || `Step ${stepNumber}: ${actionLabel(action, data.value)}`;
      addConsoleLine("ACT", `[Step ${stepNumber}] ${action.toUpperCase()} | conf=${data.confidence?.toFixed(2)} | "${(data.caption||"").slice(0,50)}"`);

      // G. EXECUTE
      await executeAction(data);

      // Record completed action in stepHistory (sliding window of max 5 steps)
      stepHistory.push({
        step:    stepNumber,
        action:  action,
        value:   data.value || "",
        caption: data.caption || ""
      });
      if (stepHistory.length > 5) {
        stepHistory.shift(); // delete the 1st step if 6th step occurs
      }

      // H. STOP CONDITIONS
      if (action === "stop" || data.is_goal_achieved || repeatCount >= 5) {
        markLastChipDone();
        const msg = data.caption || data.value || "✅ Task complete!";
        output.textContent = msg;
        updateStatus("✅", "Goal achieved!", stepNumber);
        addConsoleLine("INFO", `Task finished in ${stepNumber} steps`);
        if (voiceEnabled) speak(msg);
        isRunning = false;
        break;
      }

      // I. ADVANCE STATE
      prevCtx = context;
      lastAct = action;
      await sleep(2500);

    } catch (err) {
      console.error("Agent error:", err);
      output.textContent = `❌ Error: ${err.message}`;
      updateStatus("❌", `Error: ${err.message.slice(0,40)}`);
      addConsoleLine("ERROR", err.message);
      statBadge.className   = "badge error";
      statBadge.textContent = "ERROR";
      isRunning = false;
      break;
    }
  }

  setUIBusy(false);
}

// ═══════════════════════════════════════════════════════════
//  4. ACTION EXECUTOR
// ═══════════════════════════════════════════════════════════
async function executeAction(data) {
  const action = (data?.action || "").trim().toLowerCase();
  const { selector, value, caption } = data;

  if (value && (value.startsWith("chrome://") || value.startsWith("chrome-extension://"))) return;

  try {
    switch (action) {
      case "open_new_tab":
      case "navigate":
        await chrome.tabs.create({ url: value });
        await sleep(2500);  // Wait for tab to load
        break;

      case "click":
      case "type":
      case "scroll":
      case "key":
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
        await sleep(2000);
        break;

      case "chat":
        output.textContent = value || "Agent response received.";
        if (voiceEnabled) speak(output.textContent);
        break;

      case "coding":
        output.textContent = `💻 Coding task: ${value || "generating code…"}`;
        await runScript("type", data);
        break;
    }
  } catch (e) {
    console.error("Executor:", e);
    addConsoleLine("ERROR", `Executor: ${e.message}`);
  }
}

// ── DOM SCRIPT ─────────────────────────────────────────────
async function runScript(type, data) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;

  await chrome.scripting.executeScript({
    target: { tabId: tab.id, allFrames: true },
    args:   [type, data],
    func: (action, params) => {
      function isVisible(el) {
        if (!el) return false;
        if (el.type === 'hidden') return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      }

      function getDeepElements(selector, root = document) {
        let list = [...root.querySelectorAll(selector)];
        const elements = root.querySelectorAll('*');
        for (const el of elements) {
          if (el.shadowRoot) {
            list = list.concat(getDeepElements(selector, el.shadowRoot));
          }
        }
        return list;
      }

      function findEl() {
        if (params.selector) {
          const els = getDeepElements(params.selector).filter(isVisible);
          if (els.length) return els[0];
        }

        if (params.value && ["click","type"].includes(action)) {
          const low = params.value.toLowerCase();
          const allEls = getDeepElements("button,a,div,span,input,textarea,label,p,h1,h2,h3,h4,li,option");
          const found = allEls
            .filter(isVisible)
            .find(e => (e.innerText || e.value || "").toLowerCase().includes(low));
          if (found) return found;
        }
        
        if (action === "type") {
          const inputs = getDeepElements("textarea,input:not([type='hidden']),[contenteditable='true']")
                .filter(isVisible);
          if (inputs.length) return inputs[0];
        }
        return null;
      }

      function pressKey(el, key) {
        const kc = key === "Enter" ? 13 : key.charCodeAt(0);
        const opts = { key, code: key, keyCode: kc, which: kc, bubbles: true, cancelable: true };
        el.dispatchEvent(new KeyboardEvent("keydown",  opts));
        el.dispatchEvent(new KeyboardEvent("keypress", opts));
        el.dispatchEvent(new KeyboardEvent("keyup",    opts));
        
        if (key === "Enter") {
          const form = el.form || el.closest("form");
          if (form) {
            try {
              if (typeof form.requestSubmit === "function") {
                form.requestSubmit();
              } else {
                form.submit();
              }
            } catch (e) {
              form.submit();
            }
          }
        }
      }

      if (action === "key") {
        const active = document.activeElement || document.querySelector("input:focus,textarea:focus,textarea,input");
        if (active) pressKey(active, params.value || "Enter");
        return;
      }

      if (action === "scroll") { window.scrollBy(0, params.value === "up" ? -500 : 500); return; }

      const el = findEl();
      if (!el) return;

      try {
        if (action === "click") {
          el.focus(); el.click(); return;
        }
        if (action === "type") {
          el.focus();
          if (el.classList?.contains("CodeMirror") && el.CodeMirror) {
            el.CodeMirror.setValue(""); el.CodeMirror.setValue(params.value);
          } else if (el.contentEditable === "true") {
            el.innerText = params.value;
          } else {
            el.value = ""; el.value = params.value;
          }
          el.dispatchEvent(new Event("input",  { bubbles: true }));
          el.dispatchEvent(new Event("change", { bubbles: true }));
          // Auto-Enter for search fields
          const isSearch = ["search","q"].some(k =>
            (params.selector||"").includes(k) ||
            (el.name||"").includes(k) ||
            (el.id||"").includes(k)
          ) || el.type === "search";
          if (isSearch) setTimeout(() => pressKey(el, "Enter"), 350);
        }
      } catch (_) {}
    }
  });
}

// ═══════════════════════════════════════════════════════════
//  5. LIVE CONSOLE
// ═══════════════════════════════════════════════════════════
function msgClass(msg) {
  if (msg.includes("[CRITICAL]")) return "cmsg-crit";
  if (msg.includes("[ERROR]"))    return "cmsg-err";
  if (msg.includes("[ACTION]"))   return "cmsg-act";
  if (msg.includes("[LLM]"))      return "cmsg-llm";
  return "cmsg-info";
}

function addConsoleLine(level, text) {
  const time = new Date().toLocaleTimeString("en-GB");
  const cls  = msgClass(`[${level}]`);
  const line = document.createElement("div");
  line.className = "cline";
  line.innerHTML = `<span class="ctime">[${time}]</span><span class="${cls}">[${level}] ${escHtml(text)}</span>`;
  consoleLines.appendChild(line);
  if (consoleLines.children.length > 80) consoleLines.removeChild(consoleLines.firstChild);
  consoleLines.scrollTop = consoleLines.scrollHeight;
}

async function fetchServerLogs() {
  try {
    const r    = await fetch(`${API}/api/logs`);
    const logs = await r.json();
    if (logs.length === lastLogCount) return;

    const newLogs = logs.slice(lastLogCount);
    lastLogCount  = logs.length;

    newLogs.forEach(l => {
      const cls  = msgClass(l.msg);
      const line = document.createElement("div");
      line.className = "cline";
      line.innerHTML = `<span class="ctime">[${l.time}]</span><span class="${cls}">${escHtml(l.msg)}</span>`;
      consoleLines.appendChild(line);
    });

    if (newLogs.length) consoleLines.scrollTop = consoleLines.scrollHeight;
    if (consoleLines.children.length > 100) {
      while (consoleLines.children.length > 80) consoleLines.removeChild(consoleLines.firstChild);
    }
  } catch (_) {}
}

function startLogPolling(fast) {
  stopLogPolling();
  logPollTimer = setInterval(fetchServerLogs, fast ? 1000 : 3000);
}
function stopLogPolling() {
  if (logPollTimer) { clearInterval(logPollTimer); logPollTimer = null; }
}

// ═══════════════════════════════════════════════════════════
//  6. VOICE
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
  const synth  = window.speechSynthesis;
  const u      = new SpeechSynthesisUtterance(text);
  const voices = synth.getVoices();
  u.voice = voices.find(v => v.name.includes("Google") && v.lang.startsWith("en")) || voices[0];
  synth.cancel();
  synth.speak(u);
}

micBtn.addEventListener("click", async () => {
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));

  if (recognition) {
    recognition.stop(); recognition = null;
    micBtn.classList.remove("rec");
    micBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    return;
  }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { output.textContent = "⚠️ Speech not supported in this browser."; return; }

  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true });
    s.getTracks().forEach(t => t.stop());
  } catch (_) {}

  recognition = new SR();
  recognition.lang = "en-US";
  recognition.continuous = false;
  recognition.interimResults = true;

  micBtn.classList.add("rec");
  micBtn.innerHTML = '<i class="fas fa-circle"></i>';
  updateStatus("🎙️", "Listening…");

  recognition.onresult = e => {
    const last = e.results[e.results.length - 1];
    const txt  = normalise(last[0].transcript);
    taskInput.value = txt;
    if (last.isFinal) {
      recognition.stop(); recognition = null;
      micBtn.classList.remove("rec");
      micBtn.innerHTML = '<i class="fas fa-microphone"></i>';
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
      addConsoleLine("INFO", `Voice: "${txt}"`);
      runAgent(txt);
    }
  };
  recognition.onerror = e => {
    recognition = null;
    micBtn.classList.remove("rec");
    micBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    addConsoleLine("ERROR", `Mic error: ${e.error}`);
    updateStatus("⚠️", `Mic error: ${e.error}`);
  };
  recognition.onend = () => {
    recognition = null;
    micBtn.classList.remove("rec");
    micBtn.innerHTML = '<i class="fas fa-microphone"></i>';
  };
  recognition.start();
});

// ═══════════════════════════════════════════════════════════
//  7. HISTORY
// ═══════════════════════════════════════════════════════════
histBtn.addEventListener("click", () => {
  histOverlay.style.display = "block";
  loadHistory();
});

closeHist.addEventListener("click", () => {
  histOverlay.style.display = "none";
});

clearHistBtn.addEventListener("click", async () => {
  clearHistBtn.textContent = "Clearing…";
  try {
    await fetch(`${API}/api/history/clear`, { method: "DELETE" });
    histList.innerHTML = '<div class="hist-empty">History cleared ✓</div>';
  } catch (_) {
    histList.innerHTML = '<div class="hist-empty" style="color:var(--danger)">Server unreachable</div>';
  }
  clearHistBtn.innerHTML = '<i class="fas fa-trash"></i> Clear All History';
});

async function loadHistory() {
  histList.innerHTML = '<div class="hist-empty">Loading…</div>';
  try {
    const r    = await fetch(`${API}/history`);
    const data = await r.json();

    if (!data.length) {
      histList.innerHTML = '<div class="hist-empty">No history yet — run a task first!</div>';
      return;
    }

    histList.innerHTML = "";
    data.forEach(item => {
      const isCloud = item.model === "gemini-cloud";
      const d = document.createElement("div");
      d.className = "hist-item";
      d.innerHTML = `
        <div class="hist-cmd">${escHtml(item.command || "")}</div>
        <div class="hist-right">
          <div class="hist-meta">${item.time || ""}</div>
          <span class="hist-model-tag ${isCloud ? "cloud" : ""}">${item.model || "unknown"}</span>
          <button class="hist-del" data-id="${escHtml(item.id||"")}"><i class="fas fa-trash-can"></i></button>
        </div>
      `;
      // Click row → load command
      d.querySelector(".hist-cmd").addEventListener("click", () => {
        taskInput.value = item.command;
        histOverlay.style.display = "none";
      });
      // Delete single item
      d.querySelector(".hist-del").addEventListener("click", async e => {
        e.stopPropagation();
        const itemId = e.currentTarget.dataset.id;
        if (itemId) {
          await fetch(`${API}/api/history/${itemId}`, { method: "DELETE" }).catch(()=>{});
        }
        d.remove();
        if (!histList.children.length) {
          histList.innerHTML = '<div class="hist-empty">No history yet</div>';
        }
      });
      histList.appendChild(d);
    });
  } catch (e) {
    histList.innerHTML = `<div class="hist-empty" style="color:var(--danger)">Could not load — is the server running?</div>`;
  }
}

// ═══════════════════════════════════════════════════════════
//  8. EVENT BINDINGS
// ═══════════════════════════════════════════════════════════
runBtn.addEventListener("click", () => {
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
  runAgent(taskInput.value.trim());
});

stopBtn.addEventListener("click", () => {
  isRunning = false;
  setUIBusy(false);
  output.textContent = "⏹ Stopped by user.";
  updateStatus("⏹", "Stopped by user");
  addConsoleLine("WARN", "Task stopped by user");
  markLastChipDone();
  startLogPolling(false);
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

document.getElementById("clearOutputBtn").addEventListener("click", () => {
  output.textContent = "Output cleared.";
});

// ── Speed up log polling while running ─────────────────────
const _origRunAgent = runAgent;
// Patch: speed up console polling while agent runs
const origSetUIBusy = setUIBusy;

// Watch task start/stop to manage polling speed
const busyObserver = new MutationObserver(() => {});
runBtn.addEventListener("click", () => startLogPolling(true));
stopBtn.addEventListener("click", () => startLogPolling(false));

// ═══════════════════════════════════════════════════════════
//  9. HELPERS
// ═══════════════════════════════════════════════════════════
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function escHtml(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function savePref(k, v) {
  try { chrome.storage.local.set({ [k]: v }); } catch (_) {}
}

async function loadPrefs() {
  try {
    const p = await chrome.storage.local.get(["selectedModel","voiceEnabled"]);
    if (p.selectedModel) selectedModel = p.selectedModel;
    if (p.voiceEnabled !== undefined) {
      voiceEnabled = p.voiceEnabled;
      voiceTogBtn.className = "hbtn" + (voiceEnabled ? " act" : "");
      voiceTogBtn.innerHTML = `<i class="fas fa-headphones"></i> ${voiceEnabled ? "Voice" : "Muted"}`;
    }
  } catch (_) {}
}

// Patch runAgent to manage log polling speed
(function patchRunAgent() {
  const orig = runAgent;
  window.runAgent = async function(cmd) {
    startLogPolling(true);
    await orig(cmd);
    startLogPolling(false);
  };
})();

// ═══════════════════════════════════════════════════════════
//  10. BOOT
// ═══════════════════════════════════════════════════════════
(async () => {
  await loadPrefs();
  await loadModels();
  setInterval(loadModels, 30000);  // Refresh model availability
  startLogPolling(false);           // Start slow log poll at 3s
})();