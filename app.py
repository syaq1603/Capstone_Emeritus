import os, json, sqlite3, time, traceback
from typing import Optional, List, Tuple
from collections import defaultdict, deque

import requests
import pandas as pd
import numpy as np
import re

from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Request, FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Load .env in local dev ---
load_dotenv()

# ---------------- Config / Keys ----------------
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")  # don't crash if missing
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
INDEX_DIR        = os.getenv("INDEX_DIR", "chroma_index")  # persistent dir for Chroma
MAX_TURNS        = int(os.getenv("MAX_TURNS", "6"))
WEATHER_API_KEY  = os.getenv("WEATHER_API_KEY", "")      # optional
REPLICATE_TOKEN  = os.getenv("REPLICATE_API_TOKEN", "")  # optional

# ---------------- RAG setup (Chroma) ----------------
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Recommender encoder (native sentence-transformers)
from sentence_transformers import SentenceTransformer

_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 1) Embeddings object for LangChain / Chroma (replaces deprecated class)
emb = HuggingFaceEmbeddings(model_name=_EMB_MODEL)

# 2) Separate encoder for your recommender (fixes `_rec_model` not defined)
_rec_model = SentenceTransformer(_EMB_MODEL)

# Create/load persistent Chroma DB (empty is fine at start)
os.makedirs(INDEX_DIR, exist_ok=True)
vstore = Chroma(persist_directory=INDEX_DIR, embedding_function=emb)

def ensure_retriever():
    return vstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

retriever = ensure_retriever()

# LLM (guard if key missing)
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)

SYSTEM_RAG = (
    "You are a helpful assistant. Use only the provided context. "
    "Cite sources as [filename p.N] at the end. "
    "If the answer is not in the context, say you don't know."
)

def rag_answer(question: str, turns: List[str]) -> str:
    if not OPENAI_API_KEY:
        return "LLM is not configured (OPENAI_API_KEY missing)."

    # limited memory preamble
    preamble = ""
    if turns:
        compact = "\n".join(turns[-MAX_TURNS:])
        preamble = f"(Recent context; keep brief)\n{compact}\n---\n"

    docs = retriever.invoke(question)
    if not docs:
        return "I couldn't find this in the documents."

    ctx = []
    for d in docs:
        src = d.metadata.get("source", "?")
        pg  = d.metadata.get("page", d.metadata.get("page_number", "?"))
        ctx.append(f"{d.page_content[:900]}\n[{src} p.{pg}]")

    messages = [
        ("system", SYSTEM_RAG),
        ("user", f"{preamble}Question: {question}\n\nContext:\n" + "\n\n---\n\n".join(ctx) + "\n\nAnswer:")
    ]
    return llm.invoke(messages).content

# ---------------- Page template ----------------
def page(title: str, body_html: str) -> str:
    return f"""
<!doctype html>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui; margin: 24px; }}
  header {{ display:flex; align-items:center; gap:12px; margin-bottom:16px; }}
  header h2 {{ margin:0; font-size:1.2rem; }}
  nav a, button.back {{ padding:8px 12px; border:1px solid #ddd; border-radius:8px; background:#fafafa; cursor:pointer; text-decoration:none; }}
  .card {{ border:1px solid #eee; border-radius:12px; padding:16px; }}
</style>
<header>
  <button class="back" onclick="history.back()">← Back</button>
  <a href="/" title="Home">🏠 Home</a>
  <h2>{title}</h2>
</header>
<div class="card">
{body_html}
</div>
"""

# ---------------- Weather tools ----------------
def get_weather(location: str, unit: str = "metric") -> str:
    """
    Weather via WeatherAPI.com (https://www.weatherapi.com/)
    Env var: WEATHER_API_KEY
    unit: "metric" -> °C, anything else -> °F
    """
    key = WEATHER_API_KEY
    if not key:
        return "Weather is not configured (WEATHER_API_KEY missing)."
    try:
        r = requests.get(
            "https://api.weatherapi.com/v1/current.json",
            params={"key": key, "q": location, "aqi": "no"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        name = data["location"]["name"]
        country = data["location"]["country"]
        cond = data["current"]["condition"]["text"]
        if (unit or "").lower() == "metric":
            temp = data["current"]["temp_c"]; sym = "°C"
        else:
            temp = data["current"]["temp_f"]; sym = "°F"
        return f"{name}, {country}: {temp}{sym}, {cond}"
    except requests.HTTPError as e:
        try:
            err_json = e.response.json()
        except Exception:
            err_json = None
        if e.response is not None and e.response.status_code in (400, 401, 403):
            msg = err_json.get("error", {}).get("message") if err_json else str(e)
            return f"Weather API error: {msg}"
        return f"Weather error: {e}"
    except Exception as e:
        return f"Weather error: {e}"

def get_weather_data(location: str):
    """Return dict with {temp_c, temp_f, condition, is_hot, is_rainy} via WeatherAPI.com."""
    key = WEATHER_API_KEY
    if not key:
        return None
    try:
        r = requests.get(
            "https://api.weatherapi.com/v1/current.json",
            params={"key": key, "q": location, "aqi": "no"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        temp_c = float(data["current"]["temp_c"])
        cond = (data["current"]["condition"]["text"] or "").lower()
        is_hot = temp_c >= 30.0
        is_rainy = any(w in cond for w in ["rain", "showers", "thunder", "drizzle", "storm"])
        return {
            "location": f'{data["location"]["name"]}, {data["location"]["country"]}',
            "temp_c": temp_c,
            "temp_f": float(data["current"]["temp_f"]),
            "condition": cond,
            "is_hot": is_hot,
            "is_rainy": is_rainy,
        }
    except Exception:
        return None

# ---------------- SQL tool (demo) ----------------
import threading

DB_LOCK = threading.Lock()
# allow use across FastAPI worker threads
conn = sqlite3.connect(":memory:", check_same_thread=False)

with DB_LOCK:
    # Create table
    conn.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL,
            tags TEXT
        )
    """)

    # Seed base products
    conn.executemany(
        "INSERT INTO products (name, category, price, tags) VALUES (?,?,?,?)",
        [
            ("GlucoCheck Meter", "device", 39.9,  "glucose,meter,diabetes"),
            ("Lancet 30G",       "consumable", 6.5,"lancet,needle"),
            ("CGM Starter Kit",  "device", 199.0,  "cgm,sensor,starter"),
            ("Low-GI Brown Rice","grocery", 4.2,   "rice,lowgi,food"),
            ("Insulin Pen Case", "accessory", 12.0,"insulin,case,bag"),
        ]
    )

    # Add footwear products
    conn.executemany(
        "INSERT INTO products (name, category, price, tags) VALUES (?,?,?,?)",
        [
            ("Breathable Walking Shoes", "footwear", 59.0, "walking,shoes,outdoor"),
            ("Supportive Trainers",      "footwear", 89.0, "shoes,exercise,walking"),
        ]
    )

    # Improve footwear tags
    conn.execute("UPDATE products SET tags = COALESCE(tags,'') || ',breathable' WHERE category='footwear' AND name LIKE '%Breathable%'")
    conn.execute("UPDATE products SET tags = COALESCE(tags,'') || ',waterproof' WHERE category='footwear' AND name LIKE '%Trainer%'")

    conn.commit()

# ---------------- Recommender vectors (single build at startup) -------
with DB_LOCK:
    _prod = pd.read_sql_query("SELECT * FROM products", conn)

_prod["text"] = (
    _prod["name"].fillna("") + " | " +
    _prod["category"].fillna("") + " | " +
    _prod["tags"].fillna("")
)

# L2-normalize inside encode
_prod_vecs = _rec_model.encode(
    _prod["text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

def sql_query(q: str) -> str:
    if not q.strip().lower().startswith("select"):
        return "Only SELECT queries are allowed."
    try:
        with DB_LOCK:
            df = pd.read_sql_query(q, conn)
        return "No rows." if df.empty else df.to_markdown(index=False)
    except Exception as e:
        return f"SQL error: {e}"

def _tag_list(s: str) -> List[str]:
    return [t.strip().lower() for t in (s or "").split(",") if t.strip()]

def recommend(query: str, k: int = 3, weather: Optional[dict] = None) -> str:
    q = query
    if q.lower().startswith("recommend"):
        q = q[len("recommend"):].strip(": ").strip()

    # base similarity (normalize here for consistency)
    qv = _rec_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    sims = (_prod_vecs @ qv.T).ravel()

    # gentle weather-aware bonus
    bonus = np.zeros_like(sims)
    if weather is not None:
        is_hot = bool(weather.get("is_hot"))
        is_rainy = bool(weather.get("is_rainy"))
        for i, tags in enumerate(_prod["tags"].fillna("").tolist()):
            tags_l = _tag_list(tags)
            if is_hot and ("breathable" in tags_l or "ventilated" in tags_l or "mesh" in tags_l):
                bonus[i] += 0.06
            if is_rainy and ("waterproof" in tags_l or "water-resistant" in tags_l or "gore-tex" in tags_l):
                bonus[i] += 0.06

    final = sims + bonus
    top = np.argsort(-final)[:k]
    rows = _prod.iloc[top][["id","name","category","price","tags"]].copy()
    rows["score"] = [round(float(final[i]),3) for i in top]

    tip = ""
    if weather is not None:
        t = weather.get("temp_c")
        cond = weather.get("condition","")
        if weather.get("is_hot"):
            tip = f"Tip: consider *breathable* footwear — current temp {t:.0f} °C."
        elif weather.get("is_rainy"):
            tip = f"Tip: consider *waterproof* footwear — weather: {cond}."
        else:
            tip = f"Current weather: {t:.0f} °C, {cond}."

    table = rows.to_markdown(index=False)
    return table + ("\n\n" + tip if tip else "")

# ---------------- Text-to-Image (Replicate) ----------------
def t2i(prompt: str) -> str:
    token = REPLICATE_TOKEN
    if not token:
        return "Text-to-image is not configured (REPLICATE_API_TOKEN missing)."
    try:
        import replicate
        os.environ["REPLICATE_API_TOKEN"] = token

        MODEL = "black-forest-labs/flux-dev"  # widely available

        out = replicate.run(
            MODEL,
            input={"prompt": prompt, "width": 768, "height": 768, "num_outputs": 1}
        )

        if isinstance(out, str):
            urls = [out]
        elif isinstance(out, (list, tuple)):
            urls = list(out)
        else:
            try:
                urls = list(out)
            except Exception as e:
                return f"Image generation returned unexpected type: {type(out).__name__}: {e}"

        if not urls:
            return "Image generation returned no outputs."
        return "Image: " + ", ".join(urls)

    except Exception as e:
        return f"Image generation error: {e}"

# ---------------- Glucose triage helpers + function ----------------
def _infer_unit_and_to_mgdl(val: float, unit: Optional[str]) -> Tuple[float, str]:
    """If unit missing: infer mmol/L if 2.5–25; else mg/dL. Return (mg/dL, chosen_unit)."""
    if unit:
        unit = unit.lower().replace(" ", "")
    if unit in ("mmol/l", "mmol", "mmoll", "mmol-per-l", "mmol_l"):
        return val * 18.0, "mmol/L"
    if unit in ("mg/dl", "mgdl", "mg_per_dl", "mgdl"):
        return val, "mg/dL"
    if 2.5 <= val <= 25:  # infer mmol/L range
        return val * 18.0, "mmol/L (inferred)"
    return val, "mg/dL (inferred)"

SYMPTOM_WORDS = {
    "hypo": ["shaky", "sweaty", "sweating", "palpitations", "blurred", "confused", "confusion", "dizzy", "dizziness", "hungry", "tingling"],
    "dka":  ["nausea", "vomit", "vomiting", "abdominal", "stomach", "breathing", "rapid breathing", "kussmaul", "fruity", "confused", "confusion", "drowsy"]
}

def _context_flags(text: str) -> dict:
    t = text.lower()
    return {
        "fasting": any(w in t for w in ["fasting", "before breakfast", "prebreakfast", "pre-meal", "pre meal", "before meal"]),
        "postmeal": any(w in t for w in ["after meal", "post meal", "post-meal", "after lunch", "after dinner", "1h", "2h", "postprandial"]),
        "ketones": any(w in t for w in ["ketone", "ketones", "ketonemia", "ketonuria"]),
        "dka_symptoms": any(w in t for w in SYMPTOM_WORDS["dka"]),
        "hypo_symptoms": any(w in t for w in SYMPTOM_WORDS["hypo"]),
        "says_high": any(w in t for w in ["high", "spike", "spiked"]),
        "says_low": any(w in t for w in ["low", "hypo"]),
    }

def _classify_one(mgdl: float, ctx: dict) -> Tuple[str, List[str]]:
    """Return (level, bullets). level ∈ {'RED','AMBER','GREEN'}."""
    bullets: List[str] = []
    if mgdl < 54:
        return "RED", ["Severe low (<54 mg/dL / 3.0 mmol/L). Treat immediately with fast-acting carbs and seek urgent help."]
    if mgdl >= 300 and (ctx["ketones"] or ctx["dka_symptoms"]):
        return "RED", ["Very high with ketones/symptoms. Risk of DKA — seek urgent care now."]
    if mgdl < 70:
        bullets.append("Low (<70 mg/dL / 3.9 mmol/L). Use the 15/15 rule and recheck in 15 minutes.")
        return "AMBER", bullets
    if mgdl >= 300:
        bullets.append("Very high (≥300 mg/dL / 16.7 mmol/L). Hydrate, recheck, and follow your care plan; check ketones if instructed.")
        return "AMBER", bullets
    if mgdl >= 250:
        bullets.append("High (≥250 mg/dL / 13.9 mmol/L). Recheck, hydrate; consider ketone check if advised by your team.")
        return "AMBER", bullets
    if 70 <= mgdl < 180:
        bullets.append("In a common outpatient range; confirm your personal targets with your care team.")
        return "GREEN", bullets
    bullets.append("Above typical targets; consider rechecking and following your plan.")
    return "AMBER", bullets

def _target_hint(mgdl: float, ctx: dict) -> Optional[str]:
    if ctx["postmeal"] and 70 <= mgdl < 180:
        return "Within a common post-meal target (<180 mg/dL) for many adults."
    if ctx["fasting"] and 80 <= mgdl <= 130:
        return "Within a common pre-meal target (80–130 mg/dL) for many adults."
    return None

def triage_glucose_message(text: str) -> str:
    """
    Patient-safe triage:
      - Parses multiple values like '5.6 mmol, 8.2 after dinner, 210 mg/dL today'
      - Infers units if missing
      - Uses context (fasting / post-meal / symptoms / ketones)
      - Returns RED/AMBER/GREEN with guidance
    """
    ctx = _context_flags(text)
    readings: List[Tuple[float, str, float]] = []
    for m in re.finditer(r'(?P<val>\d+(\.\d+)?)\s*(?P<unit>mmol/?l|mg/?dl)?', text.lower()):
        val = float(m.group("val"))
        unit = m.group("unit")
        mgdl, chosen = _infer_unit_and_to_mgdl(val, unit)
        readings.append((val, chosen, mgdl))
    readings = [r for r in readings if 20 <= r[2] <= 1000]
    if not readings:
        if ctx["says_low"] or ctx["hypo_symptoms"]:
            return ("Possible low glucose based on your message. If you suspect a low:\n"
                    "• Take 15 g fast-acting carbs (e.g., glucose tabs/juice)\n"
                    "• Recheck in 15 minutes; repeat if still <70 mg/dL\n"
                    "• Contact your care team if lows are frequent\n\n"
                    "Tip: include a number next time, e.g., 'glucose: 3.6 mmol/L before lunch'")
        if ctx["says_high"] or ctx["dka_symptoms"] or ctx["ketones"]:
            return ("Possible high glucose based on your message. If you have ketones or feel unwell (nausea, vomiting, rapid breathing, confusion), seek urgent care.\n"
                    "Otherwise hydrate, recheck, and follow your plan.\n\n"
                    "Tip: include a number next time, e.g., 'glucose: 16.5 mmol/L with ketones'")
        return ("I didn’t detect a glucose number. Try e.g., 'glucose: 5.8 mmol/L fasting' or 'glucose: 110 mg/dL after dinner'.")

    lines: List[str] = []
    overall_level_rank = {"RED": 3, "AMBER": 2, "GREEN": 1}
    worst_level = "GREEN"

    for (raw, chosen_unit, mgdl) in readings:
        level, bullets = _classify_one(mgdl, ctx)
        hint = _target_hint(mgdl, ctx)
        line = f"• {mgdl:.0f} mg/dL"
        if "mmol" in chosen_unit:
            line += f" ({raw:.1f} mmol/L inferred)"
        if ctx["fasting"]:
            line += " (fasting)"
        elif ctx["postmeal"]:
            line += " (post-meal)"
        lines.append(line)
        if hint: bullets.append(hint)
        for b in bullets: lines.append("  - " + b)
        if overall_level_rank[level] > overall_level_rank[worst_level]:
            worst_level = level

    header = {"RED": "🚨 RED — act now / seek urgent help",
              "AMBER": "⚠️  Amber — contact your clinician soon",
              "GREEN": "✅ Green — generally within common ranges"}[worst_level]

    footer = ("\nThis assistant is educational only and not a diagnosis. "
              "Follow your clinician’s instructions and local emergency guidance.")

    return "\n".join([header, *lines, "", footer])

# ---------------- Simple router (one endpoint) ----------------
def controller(message: str, turns: List[str]) -> str:
    raw = (message or "").strip()
    if not raw:
        return "Please type a message."
    text = raw.lower()

    # Split batch commands by ';' or newlines
    parts = [p.strip() for p in re.split(r"[;\n]+", raw) if p.strip()]

    # If the batch contains both weather and recommend, prefetch structured weather once
    weather_struct = None
    weather_part = next((p for p in parts if p.lower().startswith("weather")), None)
    if weather_part is not None and any(p.lower().startswith("recommend") for p in parts):
        # Extract clean location after 'weather:'
        m = re.search(r"^weather\s*:\s*([^\n;]+)", weather_part, flags=re.I)
        loc = (m.group(1).strip() if m else weather_part[len("weather"):].strip()) or "Singapore"
        weather_struct = get_weather_data(loc)

    outputs = []
    for part in parts:
        low = part.lower()

        # Help
        if low in ("help", "/help", "how", "instructions"):
            outputs.append(
                "You can ask:\n"
                "• Document Q&A (no prefix) — e.g., 'Summarize pre-meal glucose targets'\n"
                "• Glucose check — type a number (e.g., '5.6 mmol/L after dinner') or 'glucose: 220 mg/dL'\n"
                "• Weather — 'weather: Singapore'\n"
                "• SQL — 'sql: SELECT name, price FROM products WHERE category=\"device\"'\n"
                "• Recommender — 'recommend breathable walking shoes'\n"
                "• Image — 'image: plate method visual, A4 poster'\n"
                "Educational support only — not diagnostic."
            )
            continue

        # SQL
        if low.startswith("sql:"):
            outputs.append(sql_query(part[4:].strip()))
            continue

        # Recommender (pass weather if we preloaded it)
        if low.startswith("recommend"):
            outputs.append(recommend(part, k=3, weather=weather_struct))
            continue

        # Weather (single intent)
        if low.startswith("weather"):
            m = re.search(r"^weather\s*:\s*([^\n;]+)", part, flags=re.I)
            loc = (m.group(1).strip() if m else part[len("weather"):].strip()) or "Singapore"
            outputs.append(get_weather(loc))
            continue

        # Images
        if low.startswith(("image:", "generate an image", "create an image", "poster:", "visual:")):
            prompt = part.split(":", 1)[1].strip() if ":" in part else part
            if prompt.lower().startswith("meal plan"):
                prompt += ", 7-day diabetes-friendly meal plan calendar, clean grid, A4 portrait"
            outputs.append(t2i(prompt))
            continue

        # Glucose triage (heuristic)
        if (
            low.startswith("glucose:")
            or re.search(r'\b(\d+(\.\d+)?)\s*(mmol/?l|mg/?dl)\b', low)
            or re.search(r'\b(\d+(\.\d+)?)\b', low)
        ):
            if any(w in low for w in ["mmol", "mg/dl", "ketone", "ketones", "vomit", "nausea",
                                      "breathing", "fruity", "confus", "shaky", "sweaty", "hypo",
                                      "high", "low"]):
                outputs.append(triage_glucose_message(part))
                continue

        # Default → RAG over clinic docs
        outputs.append(rag_answer(part, turns))

    # Join multiple sub-answers with a separator (keeps things readable)
    return "\n\n---\n\n".join(outputs)

# ---------------- FastAPI app & routes ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = "default"

# small in-memory history per session
histories = defaultdict(lambda: deque([], maxlen=MAX_TURNS))

@app.post("/upload")
async def upload_docs(request: Request, files: List[UploadFile] = File(...)):
    """
    Upload PDF/TXT files. Updates Chroma and returns:
    - HTML page (with Back/Home buttons) if called from a browser form
    - JSON if called from an API client (Accept: application/json or ?json=1)
    """
    try:
        os.makedirs(INDEX_DIR, exist_ok=True)

        docs = []
        for uf in files:
            content = await uf.read()
            path = f"/tmp/{int(time.time()*1000)}_{uf.filename}"
            with open(path, "wb") as f:
                f.write(content)

            name = uf.filename.lower()
            if name.endswith(".txt"):
                docs += TextLoader(path, encoding="utf-8").load()
            elif name.endswith(".pdf"):
                docs += PyPDFLoader(path).load()
            else:
                msg = f"Unsupported file: {uf.filename}"
                return _upload_response(request, ok=False, msg=msg)

        if not docs:
            return _upload_response(request, ok=False, msg="No text extracted from the uploaded files.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            return _upload_response(
                request, ok=False,
                msg="Documents parsed but produced 0 chunks (are they image-only scans?)."
            )

        vstore.add_documents(chunks)
        try:
            if hasattr(vstore, "persist"):
                vstore.persist()
            elif hasattr(vstore, "_client") and hasattr(vstore._client, "persist"):
                vstore._client.persist()
        except Exception as pe:
            print("Persist warning:", pe)

        # Refresh retriever
        global retriever
        retriever = ensure_retriever()

        msg = f"Upload successful.<br>Files: {', '.join(uf.filename for uf in files)}<br>Chunks added: {len(chunks)}"
        return _upload_response(request, ok=True, msg=msg)

    except Exception as e:
        print("UPLOAD ERROR:\n", traceback.format_exc())
        return _upload_response(request, ok=False, msg=str(e))

def _upload_response(request: Request, ok: bool, msg: str):
    wants_json = (
        "application/json" in (request.headers.get("accept") or "") or
        request.query_params.get("json") == "1"
    )
    if wants_json:
        status = 200 if ok else 400
        return JSONResponse({"ok": ok, "message": msg}, status_code=status)

    body = f"""
    <p style="{'color:green' if ok else 'color:#a00'}">{msg}</p>
    <div style="display:flex; gap:8px; margin-top:12px;">
      <button class="back" onclick="history.back()">← Back</button>
      <a href="/" class="btn">🏠 Home</a>
      <a href="/debug/status" class="btn">📊 Status</a>
      <a href="/chat.html" class="btn">💬 Go to chat</a>
    </div>
    """
    return HTMLResponse(page("Upload Result", body), status_code=200 if ok else 400)

@app.get("/debug/status")
def debug_status():
    try:
        count = vstore._collection.count()
    except Exception:
        try:
            res = vstore._collection.get(ids=None, limit=1)
            count = len(res.get("ids", []))
        except Exception:
            count = "unknown"
    return {"ok": True, "index_dir": INDEX_DIR, "doc_count": count}

@app.get("/chat.html", response_class=HTMLResponse)
def chat_page():
    body = """
    <div style="text-align:center; margin-top:18px;">
      <h1 style="font-size:2.2em; font-weight:800; margin:0 0 6px; color:#2c3e50;">
        💬 Diabetes Clinic Assistant — Chat
      </h1>
      <p style="color:#555; margin:0 0 16px;">Ask about clinic materials, glucose readings, visuals, weather, and more.</p>
      <div style="display:flex; gap:8px; justify-content:center; margin-bottom:12px;">
        <a href="/" class="btn" style="padding:8px 12px; border:1px solid #eaeaea; border-radius:10px; background:#fff; text-decoration:none;">🏠 Home</a>
        <a href="/upload.html" class="btn" style="padding:8px 12px; border:1px solid #eaeaea; border-radius:10px; background:#fff; text-decoration:none;">📄 Upload (Staff)</a>
        <a href="/debug/status" class="btn" style="padding:8px 12px; border:1px solid #eaeaea; border-radius:10px; background:#fff; text-decoration:none;">📊 Status</a>
      </div>
    </div>

    <div style="border:1px solid #eee; border-radius:12px; padding:14px;">
      <div id='log' style='white-space:pre-wrap;border:1px solid #ddd;border-radius:10px;padding:1rem;height:58vh;overflow:auto;background:#fff;'></div>

      <div style='margin-top:12px; display:flex; gap:8px;'>
        <input id='msg' style='flex:1; padding:10px; border:1px solid #eaeaea; border-radius:10px;' placeholder='Try: glucose: 7.8 after dinner · image: plate method poster · weather: Singapore'/>
        <button id='sendBtn' onclick='send()' style='padding:10px 14px; border-radius:10px; background:#0077cc; color:#fff; border:none;'>Send</button>
        <button id='clearBtn' onclick='clearChat()' style='padding:10px 14px; background:#6c757d; color:#fff; border:none; border-radius:10px;'>Clear</button>
      </div>

      <div style="margin-top:10px; display:flex; flex-wrap:wrap; gap:8px;">
        <button class="chip" data-msg="glucose: 3.2 mmol/L before breakfast shaky" style="padding:6px 10px; border:1px solid #eaeaea; border-radius:999px; background:#fff; cursor:pointer;">Glucose: low + symptoms</button>
        <button class="chip" data-msg="7.8 after dinner" style="padding:6px 10px; border:1px solid #eaeaea; border-radius:999px; background:#fff; cursor:pointer;">Glucose: number only</button>
        <button class="chip" data-msg="Summarize pre-meal glucose targets from my docs." style="padding:6px 10px; border:1px solid #eaeaea; border-radius:999px; background:#fff; cursor:pointer;">Docs: pre-meal targets</button>
        <button class="chip" data-msg="image: 7-day diabetes-friendly meal plan calendar, clean grid, A4 portrait" style="padding:6px 10px; border:1px solid #eaeaea; border-radius:999px; background:#fff; cursor:pointer;">Image: meal plan</button>
        <button class="chip" data-msg="weather: Singapore" style="padding:6px 10px; border:1px solid #eaeaea; border-radius:999px; background:#fff; cursor:pointer;">Weather: Singapore</button>
      </div>

      <p style="color:#888; font-size:12px; margin-top:10px;">
        Educational use only; not a substitute for professional medical advice. If you have severe symptoms, follow your care plan and local emergency guidance.
      </p>
    </div>

    <script>
    const log = document.getElementById('log');
    const box = document.getElementById('msg');

    function append(html){
      log.insertAdjacentHTML('beforeend', html);
      log.scrollTop = log.scrollHeight;
    }

    function linkify(text){
      return text.replace(/(https?:\\/\\/\\S+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
    }

    function cleanUrl(u){
      return u.replace(/[),.;\\]\\}]+$/,'');
    }

    function renderReply(reply){
      let html = linkify(reply);
      const raw = (reply.match(/https?:\\/\\/\\S+/g) || []);
      const urls = [...new Set(raw.map(cleanUrl))];
      const imgUrls = urls.filter(u => /\\.(png|jpg|jpeg|webp|gif)$/i.test(u) || u.includes('replicate.delivery'));
      if (imgUrls.length){
        html += '<br>' + imgUrls
          .map(u => `<img src="${u}" style="max-width:260px;border:1px solid #eee;border-radius:8px;margin:6px 6px 0 0">`)
          .join('');
      }
      return html;
    }

    async function send(){
      const m = box.value.trim();
      if(!m) return;
      append(`You: ${m}<br>`);
      box.value = '';
      try{
        const r = await fetch('/chat', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({message:m})
        });
        const j = await r.json();
        append(`Assistant: ${renderReply(j.reply)}<br><br>`);
      }catch(e){
        append(`Error: ${e}<br><br>`);
      }
    }

    function clearChat(){
      log.textContent = '';
      box.value = '';
      box.focus();
    }

    box.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' && !ev.shiftKey) { ev.preventDefault(); send(); }
    });

    document.addEventListener('DOMContentLoaded', ()=>{
      document.querySelectorAll('.chip').forEach(chip=>{
        chip.addEventListener('click', ()=>{
          box.value = chip.getAttribute('data-msg');
          send();
        });
      });
    });
    </script>
    """
    return page("Chat", body)

@app.get("/upload.html", response_class=HTMLResponse)
def upload_page():
    body = """
    <h3>Upload diabetes docs (PDF/TXT)</h3>
    <form action="/upload" method="post" enctype="multipart/form-data" style="display:flex; gap:12px; align-items:center;">
      <input type="file" name="files" multiple accept=".pdf,.txt"/>
      <button type="submit">Upload</button>
    </form>
    <p style="color:#666;margin-top:8px;">After upload, check <a href="/debug/status">status</a> or go to the <a href="/chat.html">chat</a>.</p>
    """
    return page("Upload Documents", body)

@app.post("/chat")
def chat(inp: ChatIn):
    try:
        sid = inp.session_id or "default"
        history = histories[sid]
        # flatten limited memory into short text snippets
        flat_turns: List[str] = []
        for i in range(0, len(history), 2):
            u = history[i] if i < len(history) else ""
            a = history[i+1] if i+1 < len(history) else ""
            flat_turns.append(f"U:{u}\nA:{a}")
        reply = controller(inp.message, flat_turns)
        history.append(inp.message); history.append(reply)
        return {"reply": reply}
    except Exception as e:
        print("CHAT ERROR:\n", traceback.format_exc())
        return {"reply": f"Internal error: {e}"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/ping")
def ping():
    return {"pong": True, "app": "gpt-advisor-browser"}

@app.get("/debug/status")
def debug_status_route():
    return debug_status()

@app.get("/debug/replicate")
def debug_replicate():
    import importlib
    info = {"replicate_installed": False, "token_present": False, "token_len": 0, "token_preview": ""}
    try:
        importlib.import_module("replicate")
        info["replicate_installed"] = True
    except Exception as e:
        info["replicate_error"] = str(e)
    tok = REPLICATE_TOKEN
    info["token_present"] = bool(tok)
    info["token_len"] = len(tok)
    if tok:
        info["token_preview"] = tok[:3] + "..." + tok[-3:]
    return info

@app.get("/debug/replicate/run")
def debug_replicate_run():
    try:
        msg = t2i("test image of a blue circle icon, simple, flat")
        return {"ok": True, "result": msg}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

@app.get("/debug/routes")
def _debug_routes():
    out = []
    for r in app.routes:
        methods = sorted(list(getattr(r, "methods", set())))
        out.append({"path": getattr(r, "path", ""), "name": getattr(r, "name", ""), "methods": methods})
    return out

@app.get("/debug/whoami")
def _debug_whoami():
    return {"file": os.path.abspath(__file__)}

@app.get("/", response_class=HTMLResponse)
def root():
    body = """
    <div style="text-align:center; margin-top:24px;">
      <h1 style="font-size:2.2em; font-weight:800; margin:0 0 6px; color:#2c3e50;">
        Diabetes Clinic Assistant
      </h1>
      <p style="color:#555; margin:0 0 18px;">Your friendly assistant for diabetes education and clinic support.</p>
    </div>

    <style>
      .card { border:1px solid #eee; border-radius:12px; padding:16px; }
      .btn { 
        padding: 8px 12px; 
        border-radius: 8px; 
        text-decoration: none; 
        font-weight: 600; 
        font-size: 0.95rem; 
        display: block;
        text-align: center;
      }
      .btn + .btn { margin-top: 8px; }
      .btn-blue  { background:#0077cc; color:#fff; }
      .btn-green { background:#43a047; color:#fff; }
      .btn-gray  { background:#6c757d; color:#fff; }
      .btn-wrap  { max-width:320px; margin:12px auto 0; }
    </style>

    <div class="card" style="max-width:720px; margin:0 auto;">
      <h3>For Patients</h3>
      <ul>
        <li>Ask questions about diabetes management and clinic materials.</li>
        <li>Check a glucose reading (e.g., <b>glucose: 6.2 mmol/L after dinner</b>).</li>
        <li>Generate visuals (e.g., <b>image: 7-day diabetes-friendly meal plan</b>).</li>
        <li>Check the weather (<b>weather: Singapore</b>).</li>
      </ul>

      <h3>For Clinic Staff</h3>
      <ul>
        <li>Upload PDFs/TXT handouts and guidelines (RAG powers answers).</li>
        <li>Ask document-based questions in Chat (no prefix needed).</li>
        <li>Try tools: <code>sql:</code>, <code>recommend ...</code>, <code>image: ...</code>, <code>glucose: ...</code></li>
      </ul>

      <div class="btn-wrap">
        <a href="/chat.html"    class="btn btn-blue">Open Chat</a>
        <a href="/upload.html"  class="btn btn-green">Upload Documents</a>
        <a href="/debug/status" class="btn btn-gray">System Status</a>
      </div>

      <p style="color:#888; font-size:12px; text-align:center; margin-top:12px;">
        Tip: Staff can upload clinic materials to improve answers. This assistant is educational only.
      </p>
    </div>
    """
    return page("Diabetes Clinic Assistant — Home", body)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

