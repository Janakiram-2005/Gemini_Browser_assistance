# 🤖 AI Agent Pro — Autonomous Browser AI with Local & Cloud LLMs

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-34d399?style=for-the-badge&logo=llama&logoColor=white)
![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-orange?style=for-the-badge&logo=googlechrome&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-8b5cf6?style=for-the-badge&logo=google&logoColor=white)

**An enterprise-grade autonomous browser agent that lets you control the web with plain English — powered by offline local LLMs (Gemma 2, Qwen 2.5) or Google Cloud Gemini, all switchable at the click of a button.**

[🚀 Live Demo](#running-locally) • [📖 Architecture](#architecture) • [🧩 Extension Setup](#browser-extension-setup) • [🤖 Models](#llm-models)

</div>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [LLM Models](#llm-models)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Running Locally](#running-locally)
- [Browser Extension Setup](#browser-extension-setup)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Security Notes](#security-notes)
- [Roadmap](#roadmap)

---

## Overview

**AI Agent Pro** is a full-stack autonomous browser automation system. It combines a Flask backend, a Chrome extension side panel, and a premium web dashboard to let an AI model take over your browser and execute multi-step tasks: opening tabs, clicking buttons, typing text, reading page content, submitting forms, and solving coding challenges — all from a single natural language instruction.

Unlike cloud-only solutions, AI Agent Pro is designed for **offline-first operation**: your data stays on your machine using Ollama-powered local LLMs, with Google Cloud Gemini available as an optional cloud fallback.

```
User Instruction ──► AI Brain (LLM) ──► JSON Action ──► Chrome Extension Executor ──► Browser
        ▲                                                                                  │
        └──────────────────── Page DOM / Screenshot feedback ──────────────────────────────┘
```

---

## Key Features

| Feature | Description |
|---|---|
| 🦙 **Local LLM Support** | Run Gemma 2 (5.4 GB) or Qwen 2.5 (397 MB) fully **offline** via Ollama |
| ☁️ **Cloud LLM Fallback** | Switch to Google Gemini 2.5 Flash (Vertex AI) with one click |
| 🔀 **Live Model Toggle** | Swap AI brains in real-time from the web UI or Chrome extension |
| 🌐 **Autonomous Browsing** | Agent opens tabs, clicks, types, scrolls, reads page content, submits forms |
| 🗂️ **Dynamic Subtask Roadmap** | Automatically divides complex goals into sequential subtasks, showing them live in the browser widget as Completed, Active (with animated spin), or Pending |
| 🧠 **Sliding Episodic Memory** | Retains a sliding buffer of the past 5 completed actions in static memory to completely eliminate recursive navigation loops |
| 🕳️ **Shadow DOM Traversal** | Recursively traverses `.shadowRoot` boundaries of custom components, capturing hidden questions, buttons, and radio selectors |
| 🔄 **Multi-Frame Iframe Support** | Captures via `allFrames: true` and merges contexts from all nested iframes (e.g., Vantage container widgets) |
| ⚡ **Form Submit Fallback** | Employs visibility filters to bypass hidden inputs and standard form submission (`requestSubmit()`) to ensure Enter key navigations always fire |
| 🚀 **Tailored Execution Modes** | Unique, specialized planning contexts in `app.py` for Execute, Read, Scan, Write, Search & Execute, and Coding tasks |
| 💻 **Coding Mode** | Detects LeetCode / coding tasks, generates & pastes code, auto-submits |
| 🎙️ **Voice Input** | Speak commands in English, Telugu, Hindi, or Tamil |
| 🔊 **Voice Output** | AI reads results aloud using Web Speech API |
| 📟 **Live Console** | Real-time color-coded backend logs streamed to the UI |
| 📂 **Command History** | MongoDB Atlas-backed history with per-command model tracking |
| 🔒 **Privacy First** | Local models never send data to the cloud |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                        │
│                                                                     │
│   ┌───────────────────────┐        ┌──────────────────────────┐    │
│   │  Web Dashboard        │        │  Chrome Extension        │    │
│   │  (templates/index.html│        │  (extension/sidepanel)   │    │
│   │                       │        │                          │    │
│   │  • Model selector grid│        │  • Model card toggle     │    │
│   │  • Voice in/out       │        │  • Task input + modes    │    │
│   │  • Live console       │        │  • History overlay       │    │
│   │  • Status pill        │        │  • Voice on/off persisted│    │
│   └──────────┬────────────┘        └────────────┬─────────────┘    │
│              │  HTTP POST /agent                 │                  │
└──────────────┼───────────────────────────────────┼──────────────────┘
               │                                   │
┌──────────────▼───────────────────────────────────▼──────────────────┐
│                        FLASK BACKEND  (app.py)                      │
│                                                                     │
│   ┌─────────────────┐    ┌────────────────┐    ┌─────────────────┐ │
│   │  /agent         │    │  /api/models   │    │  /api/logs      │ │
│   │  Route requests │    │  Live model    │    │  Stream backend │ │
│   │  to LLM router  │    │  availability  │    │  console logs   │ │
│   └────────┬────────┘    └────────────────┘    └─────────────────┘ │
│            │                                                        │
│   ┌────────▼────────────────────────────────────────────────────┐  │
│   │                    LLM ROUTING LAYER                        │  │
│   │                                                             │  │
│   │   model = "gemma2"  ──► call_ollama("gemma2:latest")        │  │
│   │   model = "qwen2.5" ──► call_ollama("qwen2.5:0.5b")        │  │
│   │   model = "gemini-cloud" ──► call_vertex("gemini-2.5-flash")│  │
│   └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
               │                              │
    ┌──────────▼──────────┐      ┌────────────▼────────────────┐
    │  Ollama (Local)     │      │  Google Vertex AI (Cloud)   │
    │  localhost:11434    │      │  us-central1                │
    │                     │      │                             │
    │  • gemma2:latest    │      │  • gemini-2.5-flash         │
    │  • qwen2.5:0.5b     │      │  • Service Account Auth     │
    └─────────────────────┘      └─────────────────────────────┘
```

---

## LLM Models

| Model | Type | Size | Speed | Best For |
|---|---|---|---|---|
| **Gemma 2** (`gemma2:latest`) | 🟢 Local / Offline | 5.4 GB | Medium | Complex reasoning, long tasks |
| **Qwen 2.5** (`qwen2.5:0.5b`) | 🟢 Local / Offline | 397 MB | ⚡ Very Fast | Quick tasks, low-resource machines |
| **Gemini 2.5 Flash** | ☁️ Cloud | N/A | Fast | Maximum intelligence, coding |

All local models run via **Ollama** at `http://localhost:11434`. The backend probes Ollama every 30 seconds to show real-time model availability in the UI.

---

## Project Structure

```
GeminiWeb-Agent/
│
├── app.py                        # Flask backend — LLM routing, agent logic, API
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (NOT committed)
├── .gitignore                    # Blocks secrets, credentials, caches
│
├── templates/
│   └── index.html                # Premium web dashboard UI
│
└── extension/                    # Chrome Extension (Manifest V3)
    ├── manifest.json             # Extension config — permissions, side panel
    ├── background.js             # Service worker — opens side panel on click
    ├── sidepanel.html            # Extension side panel UI
    └── sidepanel.js              # Agent loop, model routing, voice, history
```

### Core Files Explained

#### `app.py` — The Brain
The Flask backend is the central reasoning engine. Key responsibilities:
- **LLM Router**: Detects the `model` field in each request and routes to Ollama or Vertex AI
- **`/agent` endpoint**: Receives page context + command, builds a structured prompt, calls the LLM, parses the JSON action response
- **`/api/models` endpoint**: Probes Ollama's `/api/tags` to return live availability of each model
- **Task Type Detection**: Automatically switches to a specialised coding prompt when LeetCode/code keywords are detected
- **Loop Prevention**: Tracks `repeat_count` — if the same action repeats 3+ times, the agent stops
- **History Logging**: Every command is stored in MongoDB Atlas with timestamp and model used

#### `extension/sidepanel.js` — The Executor
The extension runs an **agentic loop** in the browser:
1. Captures the current page's DOM text (`document.body.innerText`)
2. POSTs to `/agent` with the page context, command, model, and last action
3. Receives a JSON action (`click`, `type`, `scroll`, `open_new_tab`, etc.)
4. Executes the action using Chrome scripting APIs
5. Loops back to step 1 until `is_goal_achieved: true` or `action: "stop"`

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| Google Chrome | Latest | For extension |
| [Ollama](https://ollama.com) | Latest | For local LLMs |
| MongoDB Atlas | Free tier | Optional — for history |
| Google Cloud Account | — | Optional — for Gemini cloud |

### Install Ollama Models

```bash
# Install Ollama from https://ollama.com
ollama pull gemma2
ollama pull qwen2.5:0.5b
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Janakiram-2005/Gemini_Browser_assistance.git
cd Gemini_Browser_assistance
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# MongoDB Atlas (optional — for command history)
MONGODB_URI=mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/Gemini_DB

# Ollama base URL (default: localhost)
OLLAMA_BASE_URL=http://localhost:11434
```

### 4. (Optional) Google Cloud / Vertex AI Setup

If you want to use the **Gemini Cloud** model:
1. Create a Google Cloud project
2. Enable the **Vertex AI API**
3. Create a **Service Account** with `Vertex AI User` role
4. Download the JSON key and place it in the project root
5. Update `app.py` line 36: `VERTEX_SA_FILE = "your-file-name.json"`

> ⚠️ **Never commit** your service account JSON file. It is blocked by `.gitignore`.

---

## Running Locally

```bash
# 1. Make sure Ollama is running
ollama serve

# 2. Start the Flask server
python app.py
```

The server starts on:
- **Local**: `http://127.0.0.1:5000`
- **Network**: `http://<your-ip>:5000`

Open your browser and navigate to `http://127.0.0.1:5000` to see the web dashboard.

---

## Browser Extension Setup

1. Open Chrome and navigate to `chrome://extensions`
2. Enable **Developer Mode** (toggle in the top-right corner)
3. Click **"Load unpacked"**
4. Select the `extension/` folder from this project
5. Click the puzzle icon in Chrome toolbar → find **AI Agent Pro** → click **Pin**
6. Click the **AI Agent Pro** icon to open the side panel

The extension side panel connects automatically to your running Flask server at `http://127.0.0.1:5000`.

---

## API Reference

### `POST /agent`

The core reasoning endpoint. Called by both the web UI and Chrome extension.

**Request Body:**
```json
{
  "command": "Open YouTube and search for Python tutorials",
  "model": "gemma2",
  "context": "<page DOM text>",
  "previous_context": "<previous page DOM text>",
  "last_action": "open_new_tab",
  "repeat_count": 0,
  "instruction": "execute"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `command` | string | ✅ | The user's natural language goal |
| `model` | string | ✅ | One of: `gemma2`, `qwen2.5`, `gemini-cloud` |
| `context` | string | ✅ | Current page body text (max 6000 chars) |
| `previous_context` | string | ❌ | Previous page text for comparison |
| `last_action` | string | ❌ | The last executed action type |
| `repeat_count` | number | ❌ | How many times the same action has repeated |
| `instruction` | string | ❌ | Override mode: `execute`, `read`, `scan`, `write`, `coding` |

**Response:**
```json
{
  "action": "click",
  "selector": "#search-button",
  "value": "Search",
  "caption": "Clicking the search button...",
  "reason": "The search box is filled, now submitting the query",
  "confidence": 0.92,
  "is_goal_achieved": false,
  "meta": {
    "task": "normal",
    "model": "gemma2",
    "time": "2026-05-29T10:30:00"
  }
}
```

**Available Actions:**

| Action | Description |
|---|---|
| `open_new_tab` | Opens a new tab with the given URL |
| `click` | Clicks a DOM element by CSS selector or text |
| `type` | Types text into an input field |
| `scroll` | Scrolls the page up or down |
| `close_tab` | Closes the current active tab |
| `switch_tab` | Focuses the browser window |
| `wait` | Pauses for 1.5 seconds |
| `stop` | Marks task as complete and ends the loop |
| `chat` | Returns a text response to the user (no DOM action) |

---

### `GET /api/models`

Returns the list of available LLM models with live availability status.

**Response:**
```json
[
  { "id": "gemma2",       "name": "Gemma 2 (Local · Offline)",  "badge": "LOCAL", "provider": "ollama", "available": true  },
  { "id": "qwen2.5",      "name": "Qwen 2.5 (Local · Fast)",    "badge": "LOCAL", "provider": "ollama", "available": true  },
  { "id": "gemini-cloud", "name": "Gemini 2.5 Flash (Cloud)",   "badge": "CLOUD", "provider": "vertex", "available": true  }
]
```

---

### `GET /api/logs`

Returns the last 50 backend log entries for the live console.

---

### `GET /history`

Returns the last 20 commands from MongoDB Atlas, including which model processed each one.

---

## Configuration

All runtime configuration is controlled by environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `MONGODB_URI` | — | MongoDB Atlas connection string |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Path to Vertex AI service account JSON |
| `PORT` | `5000` | Flask server port |

---

## How It Works

### The Agentic Loop

```
┌─────────────────────────────────────────────────────────┐
│  User types: "Search Google for AI news and summarize"  │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Capture page DOM   │ ◄─────────────────────┐
              └──────────┬──────────┘                       │
                         │                                  │
              ┌──────────▼──────────┐                       │
              │  Build LLM prompt   │                       │
              │  (goal + context +  │                       │
              │   last action)      │                       │
              └──────────┬──────────┘                       │
                         │                                  │
              ┌──────────▼──────────┐                       │
              │  Call LLM (Gemma2/  │                       │
              │  Qwen / Gemini)     │                       │
              └──────────┬──────────┘                       │
                         │                                  │
              ┌──────────▼──────────┐                       │
              │  Parse JSON action  │                       │
              └──────────┬──────────┘                       │
                         │                                  │
              ┌──────────▼──────────┐                       │
              │  Execute in browser │                       │
              │  (click/type/scroll)│                       │
              └──────────┬──────────┘                       │
                         │                                  │
              ┌──────────▼──────────┐                       │
              │  is_goal_achieved?  │── NO ─────────────────┘
              └──────────┬──────────┘
                      YES│
              ┌──────────▼──────────┐
              │  Report result to   │
              │  user (+ voice TTS) │
              └─────────────────────┘
```

### Coding Mode

When the command contains keywords like `leetcode`, `code`, `function`, `class`, or `write code`, the agent switches to a specialised coding prompt and follows a 3-step state machine:

1. **Generate** → LLM produces working code
2. **Type** → Code is pasted into the editor (supports CodeMirror, Monaco, contenteditable)
3. **Submit** → The submit button is clicked automatically

### Model Toggle Architecture

Each API request includes a `model` field. The backend's `generate_response()` function routes the request:

```python
def generate_response(model_key, prompt):
    if model_key in OLLAMA_MODELS:
        return call_ollama(model_key, prompt)    # Local inference
    elif model_key == "gemini-cloud":
        return call_vertex(prompt)               # Cloud inference
```

The frontend fetches `/api/models` every 30 seconds to update the live status of each model card.

---

## Security Notes

> ⚠️ **Important security considerations:**

- **Never commit** service account JSON files or API keys to GitHub
- The `.gitignore` blocks all `*.json` files except `package.json`
- Store all secrets in `.env` which is also blocked by `.gitignore`
- This project uses `CORS(app, origins="*")` — for production, restrict to your domain
- The `<all_urls>` host permission in the extension gives it access to all pages — review before distributing

---

## Roadmap

- [ ] Screenshot / Vision mode (base64 image capture → multimodal LLM)
- [ ] Support for more Ollama models (Llama 3, Mistral, Phi-3)
- [ ] Task scheduling — run agent tasks at set times
- [ ] Multi-tab parallel agent execution
- [ ] Export agent workflows as reusable scripts
- [ ] Docker compose setup for one-command deployment

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ by [Janakiram](https://github.com/Janakiram-2005)

**Star ⭐ this repo if you found it useful!**

</div>
