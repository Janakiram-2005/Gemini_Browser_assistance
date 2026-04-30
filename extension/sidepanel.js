// --- GLOBAL UI ELEMENTS ---
const output = document.getElementById("output");
const taskInput = document.getElementById("taskInput");
const modeSelector = document.getElementById("modeSelector");
const micBtn = document.getElementById("micBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const stopBtn = document.getElementById("stopBtn");
const historyBtn = document.getElementById("historyBtn");
const historyView = document.getElementById("historyView");

// --- STATE VARIABLES ---
let isRunning = false;
let previousContext = "";
let lastAction = "";
let repeatCount = 0;
let isVoiceEnabled = true;

// --- 1. THE MAIN LOOP (STATE-AWARE EXECUTION ENGINE) ---
async function runAgent(command) {
    if (!command) return;
    
    // INITIALIZE BEFORE LOOP
    isRunning = true;
    previousContext = "";
    lastAction = "";
    repeatCount = 0;

    updateUI("busy");
    output.innerText = "Engine Initializing...";

    while (isRunning) {
        try {
            // A. CAPTURE CURRENT PAGE
            const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (!activeTab || !isRunning) break;

            const scriptRes = await chrome.scripting.executeScript({
                target: { tabId: activeTab.id },
                func: () => document.body.innerText
            });
            const context = scriptRes[0]?.result || "";

            // B. SEND REQUEST
            const API_URL = "http://127.0.0.1:5000/agent";
            const instruction = modeSelector.value || undefined;

            const payload = {
                command: command,
                context: context,
                previous_context: previousContext,
                last_action: lastAction,
                repeat_count: repeatCount
            };
            if (instruction) payload.instruction = instruction;

            // STATE HEARTBEAT
            console.log(`📡 SENDING STATE | lastAction: "${lastAction}" | repeatCount: ${repeatCount}`);

            const response = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (response.status === 429) {
                output.innerText = "⚠️ Rate Limit Hit. Backing off for 10s...";
                await new Promise(r => setTimeout(r, 10000));
                continue;
            }

            if (!response.ok) {
                // HANDLE OTHER ERRORS
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.value || "Backend Unavailable");
            }

            const data = await response.json();

            // C. ACTION VALIDATION (NORMALIZED)
            const action = (data?.action || "").trim().toLowerCase();
            const validActions = ["open_new_tab", "click", "type", "scroll", "close_tab", "switch_tab", "wait", "stop", "navigate", "chat"];
            
            if (!action || !validActions.includes(action)) {
                console.log("Invalid action received:", action);
                await new Promise(r => setTimeout(r, 2000));
                continue;
            }

            // D. DEBUG LOGGING
            console.log("ACTION:", action, "VALUE:", data.value, "SELECTOR:", data.selector);

            if (!isRunning) break;

            // E. LOOP DETECTION (use normalized 'action', not raw data.action)
            if (action === lastAction) {
                repeatCount++;
            } else {
                repeatCount = 0;
            }

            // G. EXECUTE ACTION (CRITICAL: DO BEFORE STOP CHECK)
            await executeAction(data);

            // H. STOP CONDITIONS
            if (data.action === "stop" || data.is_goal_achieved === true || repeatCount >= 2) {
                output.innerText = data.caption || data.value || "Task Complete ✅";
                if (isVoiceEnabled) speak(output.innerText);
                isRunning = false;
                break;
            }

            // I. UPDATE STATE
            previousContext = context;
            lastAction = action;

            // J. RATE LIMIT (MANDATORY - 3S)
            await new Promise(r => setTimeout(r, 3000));

        } catch (err) {
            console.error("Execution Error:", err);
            output.innerText = `Error: ${err.message}`;
            isRunning = false;
            break;
        }
    }
    updateUI("idle");
}

// --- 2. ACTION EXECUTION ---
async function executeAction(data) {
    // Use NORMALIZED action (trim + lowercase) for reliable matching
    const action = (data?.action || "").trim().toLowerCase();
    const { selector, value, caption } = data;
    
    output.innerText = caption || `Executing ${action}...`;
    console.log("EXECUTING:", action, value); // VERIFICATION LOG

    if (value && (value.startsWith("chrome://") || value.startsWith("chrome-extension://"))) {
        console.log("Blocked restricted URL");
        return;
    }

    try {
        switch (action) {
            case "open_new_tab":
            case "navigate":
                await chrome.tabs.create({ url: value });
                break;

            case "click":
            case "type":
            case "scroll":
                await runInteractionScript(action, data);
                break;

            case "close_tab":
                const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
                if (tab) await chrome.tabs.remove(tab.id);
                break;

            case "switch_tab":
                window.focus();
                break;

            case "wait":
                await new Promise(r => setTimeout(r, 1500));
                break;

            case "chat":
                output.innerText = value || "Agent response received.";
                if (isVoiceEnabled) speak(output.innerText);
                break;
        }
    } catch (e) {
        console.error("Executor Failed:", e);
    }
}

// --- 3. INTERACTION SCRIPT ---
async function runInteractionScript(type, data) {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) return;

    await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        args: [type, data],
        func: (action, params) => {
            function findEl() {
                let el = document.querySelector(params.selector);
                if (!el && params.value && (action === "click" || action === "type")) {
                    const lowerVal = params.value.toLowerCase();
                    const candidates = [...document.querySelectorAll("button, a, div, span, input, textarea")];
                    el = candidates.find(e => (e.innerText || e.value || "").toLowerCase().includes(lowerVal));
                }
                if (!el && action === "type") el = document.querySelector("textarea, input, [contenteditable='true']");
                return el;
            }

            const el = findEl();
            if (!el && action !== "scroll") return;

            try {
                switch (action) {
                    case "click":
                        el.click();
                        break;
                    case "type":
                        el.focus();
                        if (el.value !== undefined) el.value = "";
                        el.innerText = "";
                        
                        if (el.classList && el.classList.contains('CodeMirror')) {
                            el.CodeMirror.setValue(""); 
                            el.CodeMirror.setValue(params.value);
                        } else if (el.contentEditable === "true") {
                            el.innerText = params.value;
                        } else {
                            el.value = params.value;
                        }
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        break;
                    case "scroll":
                        window.scrollBy(0, 500);
                        break;
                }
            } catch (e) {}
        }
    });
}

// --- 4. VOICE FIX ---
function normalizeCommand(text) {
    return text.toLowerCase()
        .replace("lead code", "leetcode")
        .replace("open mail", "open gmail")
        .replace("sign out", "logout")
        .trim();
}

function speak(t) {
    if (!isVoiceEnabled) return;
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(t);
    const voices = synth.getVoices();
    utterance.voice = voices.find(v => v.name.includes("Google") && v.lang.startsWith("en")) || voices[0];
    synth.cancel();
    synth.speak(utterance);
}

// --- 5. UI & EVENTS ---
function updateUI(state) {
    const isBusy = state === "busy";
    analyzeBtn.style.display = isBusy ? "none" : "block";
    stopBtn.style.display = isBusy ? "block" : "none";
    document.getElementById("taskBadge").innerText = isBusy ? "EXECUTING" : "IDLE";
}

function unlockAudio() {
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
}

analyzeBtn.addEventListener("click", () => { unlockAudio(); runAgent(taskInput.value.trim()); });
stopBtn.addEventListener("click", () => { isRunning = false; updateUI("idle"); output.innerText = "Stopped."; });

taskInput.addEventListener("keypress", (e) => { 
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        unlockAudio();
        runAgent(taskInput.value.trim());
    }
});

micBtn.addEventListener("click", async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(track => track.stop());
    } catch (err) {}

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;

    const recognition = new SR();
    recognition.lang = 'en-US';
    recognition.continuous = true; 
    recognition.interimResults = true; 
    
    micBtn.classList.add("listening");
    
    recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript;
        const normalized = normalizeCommand(transcript);
        taskInput.value = normalized;
        if (event.results[event.results.length - 1].isFinal) {
            unlockAudio();
            runAgent(normalized);
        }
    };
    recognition.onend = () => micBtn.classList.remove("listening");
    recognition.start();
});

document.getElementById("voiceToggle").addEventListener("click", (e) => {
    isVoiceEnabled = !isVoiceEnabled;
    e.target.classList.toggle('active', isVoiceEnabled);
    e.target.innerText = isVoiceEnabled ? "🎧 Voice" : "🔇 Muted";
    if (isVoiceEnabled) { unlockAudio(); speak("Voice enabled"); }
});

historyBtn.addEventListener("click", () => {
    const isVisible = historyView.style.display === "block";
    historyView.style.display = isVisible ? "none" : "block";
    if (!isVisible) {
        fetch("http://127.0.0.1:5000/history")
            .then(res => res.json())
            .then(data => {
                historyView.innerHTML = "";
                data.forEach(item => {
                    const div = document.createElement("div");
                    div.className = "history-item";
                    div.innerText = item.command;
                    div.onclick = () => { taskInput.value = item.command; historyView.style.display = "none"; };
                    historyView.appendChild(div);
                });
            });
    }
});