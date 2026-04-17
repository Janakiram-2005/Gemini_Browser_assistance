// --- GLOBAL UI ELEMENTS ---
const output = document.getElementById("output");
const userInput = document.getElementById("userInput");
const micBtn = document.getElementById("micBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const stopBtn = document.getElementById("stopBtn");

let abortController = null;
let isVoiceEnabled = true;
let isRunning = false;
let previousAction = "None";
let previousParams = {};
let previousPageSignature = "";
let repeatCount = 0;

function makePageSignature(text) {
    const normalized = (text || "").replace(/\s+/g, " ").trim();
    return `${normalized.slice(0, 300)}::len=${normalized.length}`;
}

// --- 1. MAIN LOOP ---
async function runAgentLoop(contextPrompt = null) {
    const text = contextPrompt || userInput.value.trim();
    if (!text) return;

    if (!isRunning) {
        isRunning = true;
        previousAction = "None";
        previousParams = {};
        previousPageSignature = "";
        repeatCount = 0;
        updateUI("busy");
        output.innerText = "Thinking...";
    }

    abortController = new AbortController();

    try {
        // A. CAPTURE STATE
        const allTabs = await chrome.tabs.query({ currentWindow: true });
        const tabList = allTabs.map(t => ({ id: t.id, title: t.title, url: t.url }));
        const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });

        let pageText = "";
        try {
            const res = await chrome.scripting.executeScript({
                target: { tabId: activeTab.id },
                func: () => document.body.innerText
            });
            pageText = res[0]?.result || "";
        } catch (e) { }
        const currentPageSignature = makePageSignature(pageText);

        // B. ASK BACKEND
        const response = await fetch("http://127.0.0.1:5000/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: text,
                tabs: tabList,
                current_text: pageText.substring(0, 4000),
                current_page_signature: currentPageSignature,
                previous_page_signature: previousPageSignature,
                last_action: previousAction,
                last_params: previousParams,
                repeat_count: repeatCount
            }),
            signal: abortController.signal
        });

        const data = await response.json();

        // C. EXECUTE ACTION
        await executeAction(data);
        const nextAction = data.action;
        const nextParams = data.params || {};
        const isSameAction = nextAction === previousAction;
        const isSameParams = JSON.stringify(nextParams) === JSON.stringify(previousParams || {});

        repeatCount = (isSameAction && isSameParams) ? repeatCount + 1 : 0;
        previousAction = nextAction;
        previousParams = nextParams;
        previousPageSignature = currentPageSignature;

        // D. LOOP DECISION
        if (data.status === "in_progress" && isRunning) {
            // Only overwrite status if it was generic
            if (output.innerText === "Thinking...") output.innerText = "Working...";
            setTimeout(() => runAgentLoop(text), 2000);
        } else {
            output.innerText = data.reply || "Task Completed ✅";
            if (isVoiceEnabled && data.reply) speak(data.reply);
            isRunning = false;
            updateUI("idle");
        }

    } catch (err) {
        if (err.name !== 'AbortError') output.innerText = "Error: " + err.message;
        isRunning = false;
        updateUI("idle");
    }
}

// --- 2. ACTION EXECUTOR ---
async function executeAction(data) {
    if (data.thought) console.log("AI Thought:", data.thought);

    if (data.action === "click_element") {
        output.innerText = `Clicking "${data.params.target_text}"...`;
        await runGenericInteraction("click", data.params);
    }
    else if (data.action === "type_text") {
        output.innerText = `Typing "${data.params.input_value}"...`;
        await runGenericInteraction("type", data.params);
    }
    else if (data.action === "scroll") {
        await runGenericInteraction("scroll", data.params);
    }
    else if (data.action === "open_url") {
        output.innerText = `Opening ${data.params?.url || data.url}...`;
        await chrome.tabs.create({ url: data.params?.url || data.url });
    }
    else if (data.action === "switch_tab") {
        output.innerText = "Switching tab...";
        await chrome.tabs.update(data.tab_id, { active: true });
        if (data.script_task) await runSkillScript(data.tab_id, data.script_task, data.params);
    }
    else if (data.action === "chat") {
        output.innerText = data.reply;
        if (isVoiceEnabled) speak(data.reply);
    }
    else if (["youtube_search", "whatsapp_send"].includes(data.action)) {
        output.innerText = `Running: ${data.action}`;
        const [active] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (active) await runSkillScript(active.id, data.action, data.params);
    }
    else if (data.action === "close_tabs") {
        output.innerText = "Closing tabs...";
        if (data.tab_id) {
            await chrome.tabs.remove(data.tab_id);
        } else {
            // Default: Close current active tab if no ID
            const [active] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (active) await chrome.tabs.remove(active.id);
        }
    }
}

// --- 3. GENERIC INTERACTION ---
async function runGenericInteraction(type, params) {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        args: [type, params],
        func: (action, p) => {
            function findElementByText(text) {
                if (!text) return null;
                const xpath = `//*[contains(text(), '${text}')] | //input[@placeholder='${text}'] | //button[contains(@aria-label, '${text}')]`;
                return document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
            }
            if (action === "click") {
                let el = p.selector ? document.querySelector(p.selector) : findElementByText(p.target_text);
                if (el) {
                    // Simulste full mouse event sequence for games/react apps
                    ['mousedown', 'mouseup', 'click'].forEach(evtType => {
                        const event = new MouseEvent(evtType, {
                            bubbles: true,
                            cancelable: true,
                            view: window,
                            buttons: 1
                        });
                        el.dispatchEvent(event);
                    });
                    el.focus();
                }
            }
            else if (action === "type") {
                let el = p.selector ? document.querySelector(p.selector) : findElementByText(p.target_text);
                if (!el) el = document.querySelector('input[type="text"], input[type="search"]');
                if (el) {
                    el.value = p.input_value;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.focus();
                }
            }
            else if (action === "scroll") {
                window.scrollBy({ top: 500, behavior: 'smooth' });
            }
        }
    });
}

// --- 4. SKILLS (FIXED YOUTUBE CLICKER) ---
async function runSkillScript(tabId, skill, params) {
    if (skill === "youtube_search") {
        chrome.scripting.executeScript({
            target: { tabId: tabId },
            args: [params.query],
            func: (q) => {
                const input = document.querySelector('input#search') || document.querySelector('[name="search_query"]');
                const btn = document.querySelector('button#search-icon-legacy') || document.querySelector('[aria-label="Search"]');

                if (input && btn) {
                    input.value = q;
                    input.dispatchEvent(new InputEvent('input', { bubbles: true }));
                    btn.click();

                    // --- NEW ROBUST CLICKER ---
                    let attempts = 0;
                    const finder = setInterval(() => {
                        attempts++;
                        // Try finding the main title link (ignores shorts shelf often)
                        const video = document.querySelector('a#video-title');
                        if (video && video.href.includes('watch')) {
                            clearInterval(finder);
                            video.click(); // CLICK IT!
                        } else if (attempts > 15) { // Stop after 7.5 seconds
                            clearInterval(finder);
                            alert("Gemini: I searched, but couldn't auto-click the video. Please click it.");
                        }
                    }, 500); // Check every 500ms
                }
            }
        });
    }
    else if (skill === "whatsapp_send") {
        chrome.scripting.executeScript({
            target: { tabId: tabId },
            args: [params.contact, params.message],
            func: (c, m) => {
                const box = document.querySelector('div[contenteditable="true"][data-tab="3"]');
                if (box) {
                    box.innerHTML = c;
                    box.dispatchEvent(new InputEvent('input', { bubbles: true }));
                    setTimeout(() => box.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })), 500);
                }
            }
        });
    }
}

// --- 5. UI SETUP ---
function updateUI(state) {
    analyzeBtn.style.display = state === "busy" ? "none" : "block";
    stopBtn.style.display = state === "busy" ? "block" : "none";
}
function speak(text) {
    if (!isVoiceEnabled) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

analyzeBtn.addEventListener("click", () => runAgentLoop());
stopBtn.addEventListener("click", () => { if (abortController) abortController.abort(); isRunning = false; });
userInput.addEventListener("keypress", (e) => { if (e.key === "Enter") runAgentLoop(); });

// --- VOICE INPUT SETUP ---
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
        micBtn.classList.add("listening");
        micBtn.innerText = "🔴";
        output.innerText = "Listening...";
    };

    recognition.onend = () => {
        micBtn.classList.remove("listening");
        micBtn.innerText = "🎙️";
        if (output.innerText === "Listening...") output.innerText = "Ready.";
    };

    recognition.onresult = (e) => {
        const transcript = e.results[0][0].transcript;
        userInput.value = transcript;
        output.innerText = `Heard: "${transcript}"`;
        setTimeout(() => runAgentLoop(), 500); // Auto-submit
    };

    recognition.onerror = (e) => {
        console.error("Speech Error:", e.error);
        micBtn.classList.remove("listening");
        micBtn.innerText = "🎙️";
        output.innerText = "Mic Error: " + e.error;
    };
} else {
    micBtn.style.display = "none";
    console.warn("Web Speech API not supported.");
}

micBtn.addEventListener("click", () => {
    if (!recognition) return alert("Voice input not supported in this browser.");
    try { recognition.start(); } catch (e) { recognition.stop(); }
});

const voiceBtn = document.getElementById("voiceToggle");
const historyBtn = document.getElementById("historyBtn");
const historyView = document.getElementById("historyView");

voiceBtn.addEventListener("click", () => {
    isVoiceEnabled = !isVoiceEnabled;
    voiceBtn.classList.toggle('active', isVoiceEnabled);
    voiceBtn.innerText = isVoiceEnabled ? "🎧" : "🔇";
});

historyBtn.addEventListener("click", () => {
    const isHidden = historyView.style.display === "none" || historyView.style.display === "";
    historyView.style.display = isHidden ? "block" : "none";
    historyBtn.classList.toggle('active', isHidden);
    if (isHidden) loadHistory();
});

async function loadHistory() {
    historyView.innerHTML = `<div style="text-align:center;color:#94a3b8;margin-top:20px;">Loading...</div>`;
    try {
        const res = await fetch("http://127.0.0.1:5000/history");
        if (!res.ok) throw new Error("Failed to load");
        const logs = await res.json();

        if (logs.length === 0) {
            historyView.innerHTML = `
                <div style="text-align:center;color:#64748b;margin-top:20px;">No history yet.</div>
                <button id="closeHistoryBtn" class="clear-all-btn" style="margin-top:10px;">Close</button>
            `;
            document.getElementById("closeHistoryBtn").addEventListener("click", () => {
                historyView.style.display = 'none';
                historyBtn.classList.remove('active');
            });
            return;
        }

        historyView.innerHTML = `
            <button id="clearAllBtn" class="clear-all-btn">🗑️ Clear All History</button>
            ${logs.map(l => `
                <div class="history-item">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-size:0.75rem;color:#94a3b8;">${l.time}</span>
                        <span class="history-status" style="font-size:0.7rem;color:${l.status === 'Success' ? '#4ade80' : '#f87171'}">${l.status}</span>
                    </div>
                    <div class="history-cmd">${l.command}</div>
                    <button class="history-del-btn" data-id="${l._id}">×</button>
                </div>
            `).join('')}
            <button id="closeHistoryBtn" class="clear-all-btn" style="margin-top:10px;border-color:#64748b;color:#94a3b8;">Close</button>
        `;

        // Attach Event Listeners
        document.getElementById("clearAllBtn").addEventListener("click", clearHistory);
        document.getElementById("closeHistoryBtn").addEventListener("click", () => {
            historyView.style.display = 'none';
            historyBtn.classList.remove('active');
        });

        document.querySelectorAll('.history-del-btn').forEach(b => {
            b.addEventListener('click', async (e) => {
                const id = e.target.getAttribute('data-id');
                if (id) {
                    await fetch(`http://127.0.0.1:5000/history/${id}`, { method: 'DELETE' });
                    loadHistory();
                }
            });
        });

    } catch (e) {
        historyView.innerHTML = `<div style="color:#ef4444;text-align:center;margin-top:20px;">Error loading history.<br>${e.message}</div>`;
    }
}

async function clearHistory() {
    if (confirm("Clear all history?")) {
        await fetch("http://127.0.0.1:5000/history", { method: 'DELETE' });
        loadHistory();
    }
}