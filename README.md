**Diabetes Clinic Assistant (Browser)**
========================================

A patient- and staff-friendly web app for diabetes education and clinic support.
It provides 
1) a conversational interface with limited memory, 
2) Document Q&A via RAG, 
3) text-to-image generation, 
4) multi-agent tools (Weather / SQL / Recommender), 
5) and glucose safety triage (educational).
------------------------------------------------------------------------------------------------------------------------------
Tech stack: FastAPI · OpenAI Chat (RAG) · Chroma (vector DB) · Sentence-Transformers (embeddings) · Replicate (images) · WeatherAPI.com (weather)

== Features ==

- Conversational interface (browser UI)
- Multi-turn chat with a rolling short-term memory.
- Inline image display in chat.
- RAG over clinic documents
- Upload PDF/TXT; assistant answers with citations.
- Multi-agent controller
- weather: <Singapore> — current conditions (WeatherAPI).
- sql: SELECT … — demo SQLite catalog (read-only).
- recommend … — content-based suggestions (MiniLM).
- image: … — educational visual generation (Replicate).
- glucose: … — non-diagnostic triage (red/amber/green).
- Fallback to RAG when no prefix is used.
- Landing page
- Clear instructions for Patients and Staff; stacked, compact buttons.
- Clinical safety: Educational only. Not a medical device.
- Patients should follow their clinician’s plan and local emergency guidance.

== System Overview ==

[ Browser UI ]  ───────────────>  [ FastAPI Controller ]
                               ┌─────────────────────────────────────────┐
                               │ Intent Router (last N turns memory)     │
                               │                                         │
                               │  RAG → Chroma + MiniLM embeddings        │
                               │  Weather → WeatherAPI.com                │
                               │  SQL → SQLite demo (read-only)           │
                               │  Recommender → MiniLM similarity         │
                               │  Text-to-Image → Replicate flux-dev      │
                               │  Glucose Triage → rules + thresholds     │
                               └─────────────────────────────────────────┘

== Quickstart (Local) ==
1) Clone & create a venv
git clone <your-repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

2) Install dependencies
pip install -r requirements.txt

If upload errors mention python-multipart:

pip install python-multipart

3) Environment variables

Create a .env in project root (or export in your shell):

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
INDEX_DIR=chroma_index
MAX_TURNS=6
PORT=9000

WEATHER_API_KEY=your_weatherapi_com_key
REPLICATE_API_TOKEN=r8_...  # replicate.com

4) Run the server
PORT=9000 python app.py
# or with hot reload:
uvicorn app:app --host 0.0.0.0 --port 9000 --reload


Open: http://127.0.0.1:9000/

== Using the App ==
Landing page (/)

= Patients = : click Open Chat, ask questions, check glucose readings, generate visuals, check weather.

= Staff = : click Upload Documents to add PDFs/TXT; ask RAG questions in chat; check System Status.

= Upload documents (/upload.html)

= Select .pdf or .txt → submit → index updates immediately.

= Chroma persists at INDEX_DIR.

= Chat (/chat.html)

Try examples:

RAG (default):
What is the plate method?
Summarize pre-meal glucose targets from my docs.

Glucose triage (educational):
glucose: 3.2 mmol/L before breakfast shaky
7.8 after dinner (unit inferred if omitted)
220 mg/dL after dinner and nauseous

Weather: Singapore

SQL (read-only): sql: SELECT name, price FROM products WHERE category='device'

Recommender: recommend a case for insulin pens

Text-to-image:
image: plate method visual for type 2 diabetes, clean infographic, A4 portrait
image: 7-day diabetes-friendly meal plan calendar, clean grid, A4 portrait

Curl samples
curl -s -X POST http://127.0.0.1:9000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"weather: Singapore"}'

curl -s -X POST http://127.0.0.1:9000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"image: plate method infographic, A4 portrait"}'

== Endpoints ==

GET / — Landing page

GET /chat.html — Chat UI (inline images, Enter-to-send, Clear)

GET /upload.html — Upload docs (PDF/TXT)

POST /chat — Chat API: {"message": "...", "session_id": "optional"}

POST /upload — Upload API (multipart form)

GET /debug/status — Vector index count & directory

GET /debug/replicate — Replicate token presence

GET /debug/replicate/run — Test image generation

GET /docs | GET /redoc — Auto API docs
===============================================================
Capstone Requirements Fulfillment 
===============================================================
== Conversational Interface ==

Criterion: Supports multi-turn, coherent dialogue with limited memory.

Browser chat at /chat.html; rolling memory:

histories = defaultdict(lambda: deque([], maxlen=MAX_TURNS))
MAX_TURNS = 6

Controller/RAG composes a compact preamble from recent turns.

Optional session_id supports multiple sessions.

== Document Querying (RAG) ==

Criterion: Users can upload documents; the assistant answers contextually using RAG.

Upload via /upload.html (PDF/TXT), chunk with RecursiveCharacterTextSplitter.

Embed using SentenceTransformerEmbeddings("all-MiniLM-L6-v2").

Persist in Chroma; retrieve top-k and answer with citations.

== Image Generation with Prompt Engineering ==

Criterion: Accepts text prompts and generates images using Replicate API. Includes prompt experimentation.

image: ... routes to t2i() → Replicate (black-forest-labs/flux-dev).

Inline rendering in chat; prompt tips: subject + style + layout + audience.

Debug routes verify token and perform test runs.

== Multi-Agent Coordination via Controller ==

Criterion: Controller manages and routes tasks to agents (Weather, SQL, Recommender). Agents collaborate for coherent results.

Controller routes by prefix or pattern:

== weather: → WeatherAPI ==

== sql: → SQLite ==

== recommend → MiniLM similarity ==

== image: → Replicate ==

== glucose: / unit patterns → Triage ==

Otherwise → RAG

== Agents can be combined in a single response (e.g., RAG + Image, SQL + RAG). ==

== Final Technical Report ==
System Design Overview

|Backend|: FastAPI app (app.py) with chat/upload/debug routes and agents.

|Agents|: RAG (Chroma + MiniLM), Weather (WeatherAPI), SQL (SQLite demo), Recommender (MiniLM), T2I (Replicate), Glucose Triage (rules).

|Memory|: Per-session rolling buffer (MAX_TURNS).

|Persistence|: Chroma index at INDEX_DIR.

++Integration Process++

Base FastAPI skeleton (chat/upload/landing).

|RAG|: loaders → splitter → embeddings → Chroma retriever.

|Controller|: keyword routing; added agents progressively.

|UI|: inline image rendering; stacked, compact buttons on landing page.

|Tokens|: .env; debug routes for status checks.

Debugging & Testing

Fixed Python 3.9 typing (Optional[...] vs | None).

Replicate test route (/debug/replicate/run) for quick diagnosis.

RAG persistence validated with /debug/status.

Frontend cache bust via hard refresh (Cmd+Shift+R).
========================
Reflections
========================
Multi-agent + RAG provides practical clinical education value.

Modularity enables future agents (appointments, medication info).

Safety: non-diagnostic framing; staff review recommended for patient-facing materials.

Lessons: dependency/version control; environment config; useful debug endpoints.

== Troubleshooting ==

Landing page didn’t change
Ensure only one @app.get("/") route exists.
Restart server; hard refresh browser.
Upload fails with python-multipart
pip install python-multipart
Weather 401
Use WeatherAPI.com key; restart server.
Image generation fails
Set REPLICATE_API_TOKEN; test GET /debug/replicate/run.
OpenAI errors
Confirm OPENAI_API_KEY; verify OPENAI_MODEL.
Vector index stuck
Ensure docs are text-based (not scanned images); check /debug/status.
=====================================
Project Structure (simplified)
=====================================
.
├── app.py                  # FastAPI app (routes, controller, agents, triage, RAG)
├── requirements.txt
├── Procfile                # (for Render/Heroku, optional)
├── chroma_index/           # (created at runtime; persistent vector store)
└── README.md

== Roadmap ==

Role-based access (patient vs staff), auth

OCR for scanned PDFs

Streaming responses; richer chat UI

Structured education datasets (CSV/JSON) as RAG sources

Multilingual support

== License & Acknowledgements ==

Built with FastAPI, Chroma, Sentence-Transformers, Replicate, and OpenAI.
Clinical targets align with common outpatient guidance; this app remains educational only.