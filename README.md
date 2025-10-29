# 🩺 Diabetes Clinic Assistant (Browser)

**A patient- and staff-friendly web app for diabetes education and clinic support.**

This prototype provides:
1. Conversational interface with limited memory  
2. Document-based Question Answering (RAG)  
3. Text-to-image generation  
4. Multi-agent tools (Weather / SQL / Recommender)  
5. Glucose safety triage (educational only)

---

### 🧰 Tech Stack
**FastAPI** · **OpenAI Chat (RAG)** · **Chroma (Vector DB)** · **Sentence-Transformers (Embeddings)** · **Replicate (Images)** · **WeatherAPI.com (Weather)**

---

## ✨ Features Overview

### 🗣 Conversational Interface
- Browser-based multi-turn chat with short-term memory (`MAX_TURNS = 6`)
- Inline image display
- Rolling session context with limited memory (privacy-safe)

### 📚 Document Q&A (RAG)
- Upload PDFs or text files  
- Assistant answers with contextual retrieval and citations  
- Persistent Chroma vector index for reuse across restarts  

### 🤖 Multi-Agent Controller
Handles multiple tasks seamlessly:
- `weather: Singapore` — fetches live conditions via WeatherAPI  
- `sql: SELECT ...` — executes read-only SQLite queries  
- `recommend ...` — content-based suggestions (MiniLM vectors)  
- `image: ...` — text-to-image via Replicate  
- `glucose: ...` — rule-based non-diagnostic triage (red/amber/green)  
- Fallback: document-grounded RAG answers  

### 💡 Clinical Safety
> **Educational use only.**  
> Not a diagnostic tool — patients should always follow clinician guidance.

---

## 🏗️ System Overview

```text
User
  │
  ▼
┌────────────────────────── Controller / Router ────────────────────────────┐
│ Uses last N turns (Short Memory: MAX_TURNS=6)                             │
│ Supports batch commands: e.g. "recommend …; weather: Singapore"           │
│                                                                            │
│  ┌────────────── RAG ──────────────┐   ┌──── Weather Agent ────┐          │
│  │ Chroma Vector Store (persistent)│   │ WeatherAPI.com         │          │
│  │  ↑ Embeddings: HuggingFace      │   └────────────────────────┘          │
│  │  │ (all-MiniLM-L6-v2)           │                                       │
│  └──┴──────────────────────────────┘   ┌────── SQL Agent ──────┐           │
│                                        │ SQLite (in-memory)     │           │
│  ┌────────── Recommender ──────────┐   │ check_same_thread=False │          │
│  │ Content-based (MiniLM vectors)  │   │ + DB_LOCK (thread-safe) │          │
│  │ (uses Weather hints optionally) │   └─────────────────────────┘          │
│  └─────────────────────────────────┘                                        │
│                                                                            │
│  ┌──────── Glucose Triage Agent ───────┐   ┌──── Image Generator ─────────┐│
│  │ RED/AMBER/GREEN (educational)       │   │ Replicate (flux-dev) + prompts││
│  └─────────────────────────────────────┘   └───────────────────────────────┘│
└────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
Final response (text + citations + optional image URL)

⚙️ Quickstart (Local)
1️⃣ Clone & Create a Virtual Environment
git clone <your-repo-url>
cd Capstone_Emeritus
python -m venv .venv
source .venv/bin/activate      # (Windows: .venv\Scripts\activate)

2️⃣ Install Dependencies
pip install -r requirements.txt
# If upload fails:
pip install python-multipart

3️⃣ Environment Variables
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_MODEL=gpt-4o-mini
INDEX_DIR=chroma_index
MAX_TURNS=6
PORT=9000

WEATHER_API_KEY=your_weatherapi_key
REPLICATE_API_TOKEN=YOUR_REPLICATE_TOKEN  # replicate.com

4️⃣ Run the Server
PORT=9000 python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 9000 --reload

💬 Using the App
🏠 Landing Page (/)

Clear “Patient” and “Staff” buttons

Staff → Upload documents

Patients → Open chat

📤 Upload Documents (/upload.html)

Accepts .pdf and .txt

Immediately updates Chroma index (INDEX_DIR)

💬 Chat (/chat.html)

Example prompts:
RAG:
"What is the plate method?"
"Summarize pre-meal glucose targets from my docs."

Glucose triage:
"glucose: 3.2 mmol/L before breakfast shaky"
"7.8 after dinner"
"220 mg/dL after dinner and nauseous"

Weather:
"weather: Singapore"

SQL:
"sql: SELECT name, price FROM products WHERE category='device'"

Recommender:
"recommend a case for insulin pens"

Text-to-Image:
"image: plate method visual, A4 portrait, clean infographic"

🧪 Curl Examples
curl -s -X POST http://127.0.0.1:9000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"weather: Singapore"}'

🔍 Endpoints Summary
| Method   | Route                  | Description           |
| :------- | :--------------------- | :-------------------- |
| **GET**  | `/`                    | Landing page          |
| **GET**  | `/chat.html`           | Chat UI               |
| **GET**  | `/upload.html`         | Upload documents      |
| **POST** | `/chat`                | Chat API              |
| **POST** | `/upload`              | File upload API       |
| **GET**  | `/debug/status`        | Vector index status   |
| **GET**  | `/debug/replicate`     | Check Replicate token |
| **GET**  | `/debug/replicate/run` | Generate test image   |
| **GET**  | `/docs`, `/redoc`      | Auto API docs         |

✅ Capstone Requirements Fulfillment
1. Conversational Interface

Multi-turn chat using deque-based rolling memory (MAX_TURNS=6).
Supports per-session memory and multiple user sessions.

2. Document Querying (RAG)

Uploads PDFs/TXT → chunked → embedded (MiniLM) → stored in Chroma.
Retrieval-based contextual responses with citations.

3. Image Generation

image: prefix triggers Replicate (black-forest-labs/flux-dev) for educational visuals.
Prompt templates include: subject + style + layout + audience.

4. Multi-Agent Coordination

Controller routes requests:

weather: → WeatherAPI

sql: → SQLite

recommend: → MiniLM recommender

image: → Replicate

glucose: → Triage

fallback → RAG

Agents can collaborate (e.g., Weather + Recommender).

5. Final Technical Report Summary

Backend: FastAPI
Agents: RAG, Weather, SQL, Recommender, Image, Glucose
Memory: Short-term deque buffer
Persistence: Chroma (persistent vector DB)

🧩 Debugging & Testing Notes
| Issue                     | Fix                                                      |
| :------------------------ | :------------------------------------------------------- |
| Landing page not updating | Remove duplicate `@app.get("/")`, restart, hard-refresh  |
| Upload fails              | `pip install python-multipart`                           |
| Weather 401               | Check WeatherAPI key                                     |
| Image generation fails    | Verify `REPLICATE_API_TOKEN`                             |
| OpenAI error              | Confirm `OPENAI_API_KEY`                                 |
| Vector index not updating | Ensure document text is extractable (not scanned images) |

🧱 Project Structure
.
├── app.py               # FastAPI app (routes, agents, controller)
├── requirements.txt
├── Procfile             # (for Render/Heroku)
├── chroma_index/        # persistent vector DB
└── README.md

🚀 Roadmap

✅ Current: Browser-based multi-agent assistant

🔜 Add CSV ingestion (structured RAG)

🔜 Role-based access (staff vs patient)

🔜 OCR for scanned PDFs

🔜 Multilingual responses

🔜 Render deployment

⚖️ License & Acknowledgements

Built with ❤️ using FastAPI, Chroma, Sentence-Transformers, Replicate, and OpenAI.
Clinical references align with public outpatient guidelines.

This assistant is for educational support only, not diagnostic use.

🙏 Acknowledgements

This project was developed as part of my Capstone Project (Emeritus Data Science & AI Program).

Special thanks to OpenAI’s ChatGPT (GPT-5) for providing technical guidance, debugging support, and writing assistance during development.

All code implementation, testing, and integration were performed independently by the author, with ChatGPT serving as a collaborative assistant for learning and documentation.




