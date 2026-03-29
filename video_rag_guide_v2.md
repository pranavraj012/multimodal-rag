# One-Day Implementation Guide: Multimodal Video RAG System
### WSL2 Ubuntu · No Docker · React Frontend · Gemini API + Ollama

---

## Recommended Test Video

**Andrej Karpathy — "The spelled-out intro to neural networks and backpropagation: building micrograd"**
→ YouTube: `https://www.youtube.com/watch?v=VMj-3S1tku0` (2.5 hrs)

**Why this video is perfect for testing:**
- Has talking-head segments, code slides, math diagrams, and whiteboard drawings → exercises ALL 4 content classifiers
- Uses concepts like "backpropagation", "gradient descent", "chain rule" → great for follow-up query testing
- Free, publicly available, well-structured with clear topic shifts
- Dense enough that students would realistically ask "show me the diagram of X" or "explain that again"

**For a shorter test (30 min), use the first 30 minutes only — download with yt-dlp:**
```bash
pip install yt-dlp
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" \
  "https://www.youtube.com/watch?v=VMj-3S1tku0" \
  -o "micrograd_lecture.mp4"
# Trim to first 30 min for faster dev testing
ffmpeg -i micrograd_lecture.mp4 -t 1800 -c copy micrograd_30min.mp4
```

---

## What SSIM Actually Is

**Structural Similarity Index Measure (SSIM)** compares two frames across three axes: luminance (brightness), contrast, and structure (local pixel patterns). It outputs a score from 0.0 (completely different) to 1.0 (identical).

In this pipeline, we compare frame N to frame N-1. When SSIM drops below the threshold (e.g., 0.85), it means the visual content changed significantly — a new slide appeared. We save the LAST frame before the drop, which is the fully-rendered version of the previous slide (not the half-transition frame).

```
Frame sequence on one slide:  f1 → f2 → f3(bullet appears) → f4(bullet appears) → f5 NEW SLIDE
SSIM scores:                  0.97   0.96        0.94                0.97           0.21 ← drop
Action:                       skip   skip        skip               SAVE f4        start new window
```

---

## Full Free Tech Stack (No Docker)

| Layer | Tool | Install | Why no-Docker works |
|---|---|---|---|
| Audio transcription | faster-whisper | pip | Fully Python, no server |
| Frame extraction | OpenCV + scikit-image (SSIM) | pip | Pure Python |
| OCR | Tesseract + pytesseract | apt + pip | Local binary |
| Vision LLM | Ollama (LLaVA) or Gemini API | install script / API key | Ollama = local binary |
| Embeddings | sentence-transformers | pip | Runs in-process |
| Vector DB (both) | ChromaDB | pip | `pip install chromadb`, persists to disk, no server needed |
| Knowledge Graph | NetworkX + JSON persistence | pip | In-memory, saved to disk per course |
| Session store | Plain dict + JSON file | stdlib | No Redis needed at this scale |
| LLM (queries) | Ollama Mistral OR Gemini API | install / key | Switchable via env var |
| Backend | FastAPI + uvicorn | pip | |
| Frontend | React + Vite | npm | |

> **Why ChromaDB over Qdrant here?** Qdrant's local mode works but still spawns an in-process Rust binary. ChromaDB is pure Python, `pip install chromadb`, persists to a folder, zero config. For this project size it's the right call. You can swap to Qdrant server later with one line change.

> **Why NetworkX over Neo4j?** Neo4j Community still requires a JVM + service. NetworkX is pure Python, we serialize the graph to a `.graphml` file per course. For a mini-project with 1-10 lectures, this is completely sufficient.

---

## WSL2 Setup (One-Time, ~20 min)

```bash
# In WSL2 Ubuntu terminal

# 1. System deps
sudo apt update && sudo apt install -y \
  python3-pip python3-venv \
  tesseract-ocr tesseract-ocr-eng \
  ffmpeg \
  nodejs npm

# 2. Python venv
cd ~
mkdir video_rag && cd video_rag
python3 -m venv venv
source venv/bin/activate

# 3. All Python deps (one command)
pip install \
  faster-whisper \
  opencv-python-headless \
  scikit-image \
  pytesseract \
  Pillow \
  sentence-transformers \
  chromadb \
  networkx \
  spacy \
  fastapi \
  uvicorn \
  python-multipart \
  google-generativeai \
  requests \
  yt-dlp

python -m spacy download en_core_web_sm

# 4. Ollama (local LLM — skip if using Gemini only today)
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &          # starts in background
ollama pull mistral     # ~4GB, for query answering
ollama pull llava       # ~4GB, for diagram description (can skip, use Gemini)

# 5. Verify Tesseract
tesseract --version
```

### Project structure
```
video_rag/
├── backend/
│   ├── stage1_ingest.py
│   ├── stage2_extract.py
│   ├── stage3_chunk.py
│   ├── stage4_index.py
│   ├── stage5_query.py
│   ├── stage6_retrieve.py
│   ├── stage7_respond.py
│   ├── stage8_analytics.py
│   ├── config.py
│   └── main.py
├── frontend/          ← React app (created with Vite)
├── data/
│   ├── chroma_db/     ← ChromaDB persists here
│   ├── graphs/        ← NetworkX graphs per course
│   └── frames/        ← Extracted keyframes
└── uploads/           ← Uploaded video files
```

---

## config.py — LLM Switching

```python
# backend/config.py
import os

# Set this to "gemini" for fast dev, "ollama" for local/free
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" | "ollama"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")     # Get free at aistudio.google.com
GEMINI_MODEL = "gemini-1.5-flash"                    # Free tier, very fast

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "mistral"
OLLAMA_VISION_MODEL = "llava"

CHROMA_PATH = "../data/chroma_db"
GRAPHS_PATH = "../data/graphs"
FRAMES_PATH = "../data/frames"
UPLOADS_PATH = "../uploads"

SSIM_THRESHOLD = 0.85      # Lower = more keyframes. Tune between 0.75-0.92
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, free, local

# LLM caller — single function, works for both providers
def call_llm(prompt: str, image_path: str = None) -> str:
    if LLM_PROVIDER == "gemini":
        return _call_gemini(prompt, image_path)
    else:
        return _call_ollama(prompt, image_path)

def _call_gemini(prompt: str, image_path: str = None) -> str:
    import google.generativeai as genai
    from PIL import Image
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    if image_path:
        img = Image.open(image_path)
        response = model.generate_content([prompt, img])
    else:
        response = model.generate_content(prompt)
    return response.text

def _call_ollama(prompt: str, image_path: str = None) -> str:
    import requests, base64
    if image_path:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        payload = {"model": OLLAMA_VISION_MODEL, "prompt": prompt,
                   "images": [img_b64], "stream": False}
    else:
        payload = {"model": OLLAMA_CHAT_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
    return r.json()["response"]
```

**To switch:** `export LLM_PROVIDER=ollama` or `export LLM_PROVIDER=gemini` before running.
Get a free Gemini key at: `https://aistudio.google.com/app/apikey`

---

## Stage 1 — Video Ingestion & Pre-processing

```python
# backend/stage1_ingest.py
import cv2
import subprocess
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from config import SSIM_THRESHOLD, FRAMES_PATH

def extract_audio(video_path: str, output_wav: str):
    """Extract audio as 16kHz mono WAV — required format for faster-whisper"""
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-ar", "16000",   # 16kHz sample rate
        "-ac", "1",        # mono channel
        "-y", output_wav   # overwrite if exists
    ], check=True, capture_output=True)

def extract_keyframes(video_path: str, lecture_id: str) -> list[dict]:
    """
    SSIM-based keyframe extraction.

    How it works:
    - Read each frame, resize to 320x180 for cheap SSIM computation
    - Compute SSIM between current frame and previous
    - SSIM < threshold means scene changed (new slide, topic shift)
    - Save the LAST frame BEFORE the drop — it's the fully-rendered slide
    - Always flush the final frame at video end (edge case)

    Returns list of: {frame_path, timestamp, frame_idx}
    """
    out_dir = Path(FRAMES_PATH) / lecture_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    keyframes = []
    prev_gray_small = None
    prev_frame = None
    prev_ts = 0.0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (320, 180))  # Small for cheap SSIM

        if prev_gray_small is not None:
            score = ssim(prev_gray_small, gray_small)

            if score < SSIM_THRESHOLD:
                # Scene change detected — save the PREVIOUS fully-rendered frame
                save_path = str(out_dir / f"kf_{frame_idx:06d}_{prev_ts:.1f}s.jpg")
                cv2.imwrite(save_path, prev_frame)
                keyframes.append({
                    "frame_path": save_path,
                    "timestamp": prev_ts,
                    "frame_idx": frame_idx - 1
                })

        prev_gray_small = gray_small
        prev_frame = frame.copy()
        prev_ts = timestamp
        frame_idx += 1

    # Always save the final frame (video may end without a scene change)
    if prev_frame is not None:
        save_path = str(out_dir / f"kf_final_{prev_ts:.1f}s.jpg")
        cv2.imwrite(save_path, prev_frame)
        keyframes.append({
            "frame_path": save_path,
            "timestamp": prev_ts,
            "frame_idx": frame_idx - 1
        })

    cap.release()
    print(f"[Stage 1] Extracted {len(keyframes)} keyframes from {frame_idx} total frames")
    return keyframes
```

---

## Stage 2 — Multimodal Feature Extraction

```python
# backend/stage2_extract.py
import cv2
import numpy as np
import pytesseract
from PIL import Image
from faster_whisper import WhisperModel
from config import call_llm

# Load Whisper once at module level (expensive to reload)
_whisper = None
def get_whisper():
    global _whisper
    if _whisper is None:
        # int8 quantization = smaller, faster, barely any accuracy loss
        _whisper = WhisperModel("base", device="cpu", compute_type="int8")
        # Use "large-v3" for best accuracy (slower). "base" is fine for dev.
    return _whisper

# ── Content-type classifier ───────────────────────────────────────────────────

def classify_frame(image_path: str) -> str:
    """
    Classify frame into one of 4 types using heuristics only (no model cost).

    Signals used:
    - talking_head: face detection present + low text density
    - text_slide:   high white background ratio + no face
    - code:         monospace font patterns (indentation) in OCR output
    - diagram:      low text, many shapes/non-uniform colors, no face
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Signal 1: Face detection (talking head)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) > 0:
        return "talking_head"

    # Signal 2: White/uniform background → text slide
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_ratio = np.sum(thresh == 255) / thresh.size
    if white_ratio > 0.65:
        # Check for code patterns via quick OCR sample
        sample = pytesseract.image_to_string(
            Image.open(image_path).resize((400, 225)),
            config="--psm 6"
        )
        lines = sample.split("\n")
        indented = sum(1 for l in lines if l.startswith("    ") or l.startswith("\t"))
        if indented >= 2:
            return "code"
        return "text_slide"

    return "diagram"

# ── Per-type feature extraction ───────────────────────────────────────────────

def extract_frame_features(image_path: str, content_type: str) -> dict:
    """Route to correct extractor. Talking heads = zero tokens, max savings."""

    if content_type == "talking_head":
        return {
            "ocr_text": "",
            "visual_description": "Talking head — instructor speaking, no slide content",
            "entities": []
        }

    elif content_type in ("text_slide", "code"):
        # Tesseract OCR — free, local, good for clean slide text
        text = pytesseract.image_to_string(
            Image.open(image_path),
            config="--psm 6 -l eng"
        ).strip()
        return {
            "ocr_text": text,
            "visual_description": "",
            "entities": []
        }

    elif content_type == "diagram":
        # LLM vision call — only for diagrams (expensive, use sparingly)
        desc = call_llm(
            prompt=(
                "This is a frame from a lecture video showing a diagram, chart, or figure. "
                "Describe what concept it illustrates in 2-3 sentences. "
                "Name any variables, arrows, boxes, or mathematical notation visible."
            ),
            image_path=image_path
        )
        return {
            "ocr_text": "",
            "visual_description": desc,
            "entities": []
        }

    return {"ocr_text": "", "visual_description": "", "entities": []}

# ── Audio transcription ───────────────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> list[dict]:
    """
    Returns list of segments: {text, start, end}
    faster-whisper with VAD filter skips silence automatically.
    """
    model = get_whisper()
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,          # Skip long silences
        vad_parameters={"min_silence_duration_ms": 500}
    )
    print(f"[Stage 2] Detected language: {info.language} ({info.language_probability:.0%})")

    result = []
    for seg in segments:
        result.append({
            "text": seg.text.strip(),
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
        })
    print(f"[Stage 2] Transcribed {len(result)} segments")
    return result

# ── Complexity score ──────────────────────────────────────────────────────────

def compute_complexity_score(ocr_text: str, content_type: str) -> float:
    """
    Predicts how likely this chunk is to confuse students.
    Used in Stage 8 for the instructor confusion dashboard.
    Range: 0.0 (simple) to 1.0 (very complex)
    """
    score = 0.0
    if content_type == "diagram":
        score += 0.3
    if content_type == "code":
        score += 0.2
    # Count math/formula indicators
    math_chars = sum(ocr_text.count(c) for c in ["∂", "∑", "∫", "→", "∇", "∈"])
    eq_count = ocr_text.count("=")
    score += min(math_chars * 0.08, 0.3)
    score += min(eq_count * 0.03, 0.2)
    return round(min(score, 1.0), 3)
```

---

## Stage 3 — Chunking & Metadata Attachment

```python
# backend/stage3_chunk.py
import spacy
nlp = spacy.load("en_core_web_sm")

FINE_CHUNK_SENTENCES = 4   # Target sentences per fine chunk

def chunk_transcript(segments: list[dict], course_id: str, lecture_id: str) -> dict:
    """
    Two-pass chunking:
    Pass 1 → fine chunks (4 sentences, force-break at segment boundary)
    Pass 2 → coarse chunk (entire lecture = one big context window)

    Fine chunks are used for precise retrieval.
    Coarse chunks are passed to LLM as broader context in Stage 7.
    """
    all_sentences = []
    for seg in segments:
        doc = nlp(seg["text"])
        for sent in doc.sents:
            all_sentences.append({
                "text": sent.text.strip(),
                "start": seg["start"],
                "end": seg["end"]
            })

    # Pass 1: fine chunks
    fine_chunks = []
    for i in range(0, len(all_sentences), FINE_CHUNK_SENTENCES):
        batch = all_sentences[i:i + FINE_CHUNK_SENTENCES]
        if not batch:
            continue
        fine_chunks.append({
            "chunk_id": f"{lecture_id}_t_fine_{i // FINE_CHUNK_SENTENCES}",
            "chunk_type": "fine",
            "course_id": course_id,
            "lecture_id": lecture_id,
            "segment_id": i // FINE_CHUNK_SENTENCES,
            "t_start": batch[0]["start"],
            "t_end": batch[-1]["end"],
            "text": " ".join(s["text"] for s in batch),
            "parent_segment_id": 0,        # All point to the one coarse chunk
            "linked_visual_chunk_ids": [],  # Filled in Stage 4
            "rewatch_count": 0,
            "query_hit_count": 0,
        })

    # Pass 2: coarse chunk (whole lecture)
    if all_sentences:
        coarse_chunk = {
            "chunk_id": f"{lecture_id}_t_coarse_0",
            "chunk_type": "coarse",
            "course_id": course_id,
            "lecture_id": lecture_id,
            "segment_id": 0,
            "t_start": all_sentences[0]["start"],
            "t_end": all_sentences[-1]["end"],
            "text": " ".join(s["text"] for s in all_sentences),
            "rewatch_count": 0,
            "query_hit_count": 0,
        }
    else:
        coarse_chunk = None

    print(f"[Stage 3] {len(fine_chunks)} fine transcript chunks, 1 coarse chunk")
    return {"fine": fine_chunks, "coarse": [coarse_chunk] if coarse_chunk else []}


def chunk_visual(keyframes: list[dict], features: list[dict],
                 course_id: str, lecture_id: str) -> dict:
    """
    Each keyframe stable window = one fine visual chunk.
    t_end = start of next keyframe (or +10s for the last one).
    """
    fine = []
    for i, (kf, feat) in enumerate(zip(keyframes, features)):
        t_end = keyframes[i + 1]["timestamp"] if i + 1 < len(keyframes) else kf["timestamp"] + 10.0
        fine.append({
            "chunk_id": f"{lecture_id}_v_fine_{i}",
            "chunk_type": "fine",
            "course_id": course_id,
            "lecture_id": lecture_id,
            "t_start": round(kf["timestamp"], 2),
            "t_end": round(t_end, 2),
            "ocr_text": feat.get("ocr_text", ""),
            "visual_description": feat.get("visual_description", ""),
            "content_type": feat.get("content_type", "unknown"),
            "complexity_score": feat.get("complexity_score", 0.0),
            "linked_transcript_chunk_ids": [],
            "rewatch_count": 0,
            "query_hit_count": 0,
        })

    print(f"[Stage 3] {len(fine)} fine visual chunks")
    return {"fine": fine}
```

---

## Stage 4 — Indexing & Storage

```python
# backend/stage4_index.py
import json
import networkx as nx
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import spacy
from config import CHROMA_PATH, GRAPHS_PATH, EMBEDDING_MODEL

# Init once at module level
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

# ChromaDB — persists to disk, no server needed
_chroma_client = None
def get_chroma():
    global _chroma_client
    if _chroma_client is None:
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client

nlp = spacy.load("en_core_web_sm")

# ── Collection setup ──────────────────────────────────────────────────────────

def get_or_create_collections():
    client = get_chroma()
    transcript_col = client.get_or_create_collection(
        name="transcript_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    visual_col = client.get_or_create_collection(
        name="visual_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    return transcript_col, visual_col

# ── Index functions ───────────────────────────────────────────────────────────

def index_transcript_chunks(chunks: list[dict]):
    col, _ = get_or_create_collections()
    embedder = get_embedder()

    ids, embeddings, metadatas, documents = [], [], [], []
    for c in chunks:
        vec = embedder.encode(c["text"]).tolist()
        ids.append(c["chunk_id"])
        embeddings.append(vec)
        documents.append(c["text"])
        # ChromaDB metadata must be str/int/float only (no lists)
        metadatas.append({
            "chunk_type": c["chunk_type"],
            "course_id": c["course_id"],
            "lecture_id": c["lecture_id"],
            "t_start": c["t_start"],
            "t_end": c["t_end"],
            "rewatch_count": c["rewatch_count"],
            "query_hit_count": c["query_hit_count"],
        })

    col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    print(f"[Stage 4] Indexed {len(chunks)} transcript chunks")


def index_visual_chunks(chunks: list[dict]):
    _, col = get_or_create_collections()
    embedder = get_embedder()

    ids, embeddings, metadatas, documents = [], [], [], []
    for c in chunks:
        # Embed the textual description of the visual content
        text = f"{c.get('visual_description', '')} {c.get('ocr_text', '')}".strip()
        if not text:
            text = "visual content"
        vec = embedder.encode(text).tolist()
        ids.append(c["chunk_id"])
        embeddings.append(vec)
        documents.append(text)
        metadatas.append({
            "chunk_type": c["chunk_type"],
            "course_id": c["course_id"],
            "lecture_id": c["lecture_id"],
            "t_start": c["t_start"],
            "t_end": c["t_end"],
            "content_type": c.get("content_type", "unknown"),
            "complexity_score": c.get("complexity_score", 0.0),
            "rewatch_count": c["rewatch_count"],
            "query_hit_count": c["query_hit_count"],
        })

    col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    print(f"[Stage 4] Indexed {len(chunks)} visual chunks")


# ── Knowledge graph (NetworkX, persisted as GraphML) ─────────────────────────

def load_graph(course_id: str) -> nx.DiGraph:
    path = Path(GRAPHS_PATH) / f"{course_id}.graphml"
    if path.exists():
        return nx.read_graphml(str(path))
    return nx.DiGraph()

def save_graph(G: nx.DiGraph, course_id: str):
    Path(GRAPHS_PATH).mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(Path(GRAPHS_PATH) / f"{course_id}.graphml"))

def extract_and_store_concepts(chunks: list[dict], course_id: str):
    """
    Extract noun phrases as concepts, build LEADS_TO edges between
    consecutive concepts in transcript. Save graph to disk.
    """
    G = load_graph(course_id)

    for chunk in chunks:
        if chunk["chunk_type"] != "fine":
            continue
        doc = nlp(chunk["text"])
        concepts = [nc.text.lower().strip() for nc in doc.noun_chunks
                    if len(nc.text.strip()) > 3]

        for concept in concepts:
            if concept not in G:
                G.add_node(concept, course_id=course_id, appearances=[])
            # Store timestamp as a JSON string (GraphML doesn't support lists)
            existing = json.loads(G.nodes[concept].get("appearances_json", "[]"))
            existing.append(chunk["t_start"])
            G.nodes[concept]["appearances_json"] = json.dumps(existing)

        # Create LEADS_TO edges between consecutive concepts in same chunk
        for i in range(len(concepts) - 1):
            a, b = concepts[i], concepts[i + 1]
            if G.has_edge(a, b):
                G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
            else:
                G.add_edge(a, b, weight=1)

    save_graph(G, course_id)
    print(f"[Stage 4] Knowledge graph: {G.number_of_nodes()} concepts, {G.number_of_edges()} edges")
```

---

## Stage 5 — Query Understanding

```python
# backend/stage5_query.py
import re, json
from dataclasses import dataclass, field
from typing import Optional
from config import call_llm

@dataclass
class RoutingObject:
    rewritten_query: str
    intent: str           # concept | visual | temporal | locator | comparison | generative
    search_targets: list  # ["transcript", "visual"] or subset
    granularity: str      # "coarse_then_fine" | "fine_only"
    temporal_anchor: Optional[dict] = None     # {t_start, t_end} if follow-up
    content_type_filter: Optional[str] = None  # "diagram" | "code" for visual queries
    generation_task: Optional[str] = None      # "quiz" | "summary"

# ── Intent routing table (deterministic, no model) ────────────────────────────

ROUTING = {
    "concept":    {"targets": ["transcript", "visual"], "granularity": "coarse_then_fine"},
    "visual":     {"targets": ["visual"],               "granularity": "fine_only"},
    "temporal":   {"targets": [],                       "granularity": "fine_only"},
    "locator":    {"targets": [],                       "granularity": "fine_only"},
    "comparison": {"targets": ["transcript", "visual"], "granularity": "coarse_then_fine"},
    "generative": {"targets": ["transcript"],           "granularity": "coarse_then_fine"},
}

# ── Regex pre-check: skip LLM rewriter if no pronouns/references ─────────────
REFERENCE_RE = re.compile(
    r"\b(that|this|it|he|she|they|those|these|again|before|after|earlier|previous|same)\b",
    re.IGNORECASE
)

def needs_rewriting(query: str, history: list) -> bool:
    return bool(REFERENCE_RE.search(query)) and len(history) > 0

def classify_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["show me", "show the", "diagram of", "chart", "figure", "image of", "picture"]):
        return "visual"
    if any(w in q for w in ["where in", "at what point", "which part of the video", "timestamp"]):
        return "locator"
    if any(w in q for w in ["what came after", "what was before", "what followed", "next after"]):
        return "temporal"
    if any(w in q for w in ["compare", "difference between", " vs ", "versus", "contrast"]):
        return "comparison"
    if any(w in q for w in ["quiz me", "test me", "summarize", "flashcard", "give me questions"]):
        return "generative"
    return "concept"

def rewrite_query(query: str, history: list) -> dict:
    """
    LLM-powered rewriter: resolves references like 'that', 'it', 'explain again'.
    Only called when pronouns detected AND history exists (saves ~80% of LLM calls).
    """
    history_str = "\n".join([
        f"Turn {i}: Q='{h['query']}' -> clip[{h.get('t_start', '?')}s-{h.get('t_end', '?')}s]"
        for i, h in enumerate(history[-5:])
    ])
    prompt = f"""You are a query rewriter for a lecture video Q&A system.
Given conversation history and the current ambiguous query, output ONLY a JSON object.

HISTORY:
{history_str}

CURRENT QUERY: "{query}"

Output JSON with these keys:
- "rewritten_query": the resolved, standalone query (no pronouns)
- "temporal_anchor": null OR {{"t_start": float, "t_end": float}} if query refers to a specific clip
- "is_followup": true/false

JSON only, no explanation:"""

    response = call_llm(prompt)
    try:
        # Strip markdown code fences if LLM adds them
        clean = response.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except:
        return {"rewritten_query": query, "temporal_anchor": None, "is_followup": False}

def build_routing_object(query: str, history: list) -> RoutingObject:
    temporal_anchor = None

    if needs_rewriting(query, history):
        result = rewrite_query(query, history)
        rewritten = result.get("rewritten_query", query)
        temporal_anchor = result.get("temporal_anchor")
    else:
        rewritten = query

    intent = classify_intent(rewritten)
    config = ROUTING[intent]

    # Visual query: check what type of visual is requested
    content_type_filter = None
    if intent == "visual":
        if any(w in rewritten.lower() for w in ["diagram", "graph", "chart", "flow", "architecture"]):
            content_type_filter = "diagram"
        elif any(w in rewritten.lower() for w in ["code", "snippet", "function", "class"]):
            content_type_filter = "code"

    return RoutingObject(
        rewritten_query=rewritten,
        intent=intent,
        search_targets=config["targets"],
        granularity=config["granularity"],
        temporal_anchor=temporal_anchor,
        content_type_filter=content_type_filter,
        generation_task="quiz" if "quiz" in rewritten.lower() else
                        "summary" if "summarize" in rewritten.lower() else None
    )
```

---

## Stage 6 — Retrieval & Re-ranking

```python
# backend/stage6_retrieve.py
import json
import networkx as nx
from pathlib import Path
from sentence_transformers import SentenceTransformer
from stage5_query import RoutingObject
from stage4_index import get_or_create_collections, get_embedder, load_graph
import spacy

nlp = spacy.load("en_core_web_sm")

def retrieve(routing: RoutingObject, course_id: str,
             session_history: list, top_k: int = 5) -> list[dict]:

    embedder = get_embedder()
    query_vector = embedder.encode(routing.rewritten_query).tolist()
    transcript_col, visual_col = get_or_create_collections()

    raw_hits = []

    # ── Transcript VDB ────────────────────────────────────────────────────────
    if "transcript" in routing.search_targets:
        where = {"$and": [
            {"course_id": {"$eq": course_id}},
            {"chunk_type": {"$eq": "fine"}}
        ]}
        results = transcript_col.query(
            query_embeddings=[query_vector],
            n_results=min(top_k * 2, 20),
            where=where,
            include=["metadatas", "documents", "distances"]
        )
        for i, (meta, doc, dist) in enumerate(zip(
            results["metadatas"][0],
            results["documents"][0],
            results["distances"][0]
        )):
            raw_hits.append({
                "source": "transcript",
                "rank": i,
                "score": 1 - dist,  # ChromaDB returns distance, convert to similarity
                "meta": meta,
                "text": doc,
                "chunk_id": results["ids"][0][i]
            })

    # ── Visual VDB ────────────────────────────────────────────────────────────
    if "visual" in routing.search_targets:
        vis_where_conditions = [{"course_id": {"$eq": course_id}}]
        if routing.content_type_filter:
            vis_where_conditions.append({"content_type": {"$eq": routing.content_type_filter}})
        vis_where = {"$and": vis_where_conditions} if len(vis_where_conditions) > 1 else vis_where_conditions[0]

        results = visual_col.query(
            query_embeddings=[query_vector],
            n_results=min(top_k * 2, 20),
            where=vis_where,
            include=["metadatas", "documents", "distances"]
        )
        for i, (meta, doc, dist) in enumerate(zip(
            results["metadatas"][0],
            results["documents"][0],
            results["distances"][0]
        )):
            raw_hits.append({
                "source": "visual",
                "rank": i,
                "score": 1 - dist,
                "meta": meta,
                "text": doc,
                "chunk_id": results["ids"][0][i]
            })

    # ── False positive filtering ──────────────────────────────────────────────
    seen_ids = {h.get("chunk_id") for h in session_history}
    filtered = [h for h in raw_hits if h["chunk_id"] not in seen_ids and h["score"] > 0.25]

    # ── Late fusion + clip scoring (temporal grouping) ────────────────────────
    clips = _fuse_into_clips(filtered)

    # ── Re-ranking: semantic (0.5) + popularity (0.3) + recency bonus (0.2) ──
    for clip in clips:
        popularity = min(clip["query_hit_count"] / 50.0, 1.0)
        # Boost clips from the beginning of the lecture for intro queries
        recency_bonus = 0.1 if clip["t_start"] < 300 else 0.0
        clip["final_score"] = (
            0.5 * clip["clip_score"] +
            0.3 * popularity +
            0.2 * recency_bonus
        )

    clips.sort(key=lambda x: x["final_score"], reverse=True)

    # ── Neo4j-equivalent: NetworkX graph enrichment ───────────────────────────
    if routing.intent in ("concept", "comparison") and clips:
        clips = _enrich_with_graph(clips, routing.rewritten_query, course_id)

    return clips[:top_k]


def _fuse_into_clips(hits: list[dict], gap_threshold: float = 30.0) -> list[dict]:
    """
    Group transcript + visual hits that are within 30s of each other.
    Take the union time range (wider window).
    HIGH confidence = both modalities agree. LOW = single modality only.
    clip_score = 0.6 * transcript_score + 0.4 * visual_score (transcript weighted higher)
    """
    clips = []
    for hit in hits:
        t_start = hit["meta"].get("t_start", 0)
        t_end = hit["meta"].get("t_end", t_start + 10)
        placed = False
        for clip in clips:
            if abs(t_start - clip["t_start"]) < gap_threshold:
                clip["t_start"] = min(clip["t_start"], t_start)
                clip["t_end"] = max(clip["t_end"], t_end)
                clip["modalities"].add(hit["source"])
                # Weighted score fusion
                if hit["source"] == "transcript":
                    clip["clip_score"] = max(clip["clip_score"], 0.6 * hit["score"])
                else:
                    clip["clip_score"] = max(clip["clip_score"], 0.4 * hit["score"])
                clip["query_hit_count"] = max(
                    clip["query_hit_count"],
                    hit["meta"].get("query_hit_count", 0)
                )
                clip["context_text"] = clip.get("context_text", "") + " " + hit["text"]
                placed = True
                break

        if not placed:
            clips.append({
                "t_start": t_start,
                "t_end": t_end,
                "modalities": {hit["source"]},
                "clip_score": hit["score"],
                "confidence": "HIGH",
                "query_hit_count": hit["meta"].get("query_hit_count", 0),
                "context_text": hit["text"],
                "chunk_ids": [hit["chunk_id"]],
                "lecture_id": hit["meta"].get("lecture_id", ""),
                "content_type": hit["meta"].get("content_type", ""),
            })

    for clip in clips:
        clip["confidence"] = "HIGH" if len(clip["modalities"]) >= 2 else "LOW"

    return clips


def _enrich_with_graph(clips: list[dict], query: str, course_id: str) -> list[dict]:
    """Add 'see also' related concepts from the knowledge graph"""
    G = load_graph(course_id)
    if G.number_of_nodes() == 0:
        return clips

    doc = nlp(query)
    noun_chunks = [nc.text.lower() for nc in doc.noun_chunks]

    see_also = []
    for concept in noun_chunks:
        if concept in G:
            neighbors = list(G.successors(concept))[:3]
            for n in neighbors:
                appearances = json.loads(G.nodes[n].get("appearances_json", "[]"))
                if appearances:
                    see_also.append({"concept": n, "t_start": appearances[0]})

    if see_also and clips:
        clips[0]["see_also"] = see_also

    return clips
```

---

## Stage 7 — Response Generation

```python
# backend/stage7_respond.py
from config import call_llm

def generate_response(routing, clips: list[dict], query: str) -> dict:
    if not clips:
        return {
            "answer": "I couldn't find relevant content for that query in this lecture.",
            "clips": []
        }

    # Build context string from top clips
    context_parts = []
    for clip in clips[:3]:
        text = clip.get("context_text", "").strip()
        if text:
            context_parts.append(f"[{clip['t_start']:.0f}s → {clip['t_end']:.0f}s]: {text[:500]}")
    context = "\n\n".join(context_parts)

    # Generate answer based on intent
    if routing.generation_task == "quiz":
        answer = call_llm(f"Generate 3 multiple choice quiz questions from this lecture content. Number them 1-3.\n\n{context}")
    elif routing.generation_task == "summary":
        answer = call_llm(f"Summarize this lecture content in exactly 3 bullet points.\n\n{context}")
    else:
        answer = call_llm(
            f"""Answer the student's question using ONLY the lecture content provided below.
Be concise (2-4 sentences). If you reference something visual, mention the timestamp.

Lecture content:
{context}

Student question: {query}
Answer:"""
        )

    return {
        "answer": answer,
        "clips": [
            {
                "t_start": c["t_start"],
                "t_end": c["t_end"],
                "confidence": c["confidence"],
                "lecture_id": c.get("lecture_id", ""),
                "content_type": c.get("content_type", ""),
                "see_also": c.get("see_also", []),
                "modalities": list(c["modalities"])
            }
            for c in clips
        ]
    }
```

---

## Stage 8 — Analytics & Feedback Loop

```python
# backend/stage8_analytics.py
from stage4_index import get_or_create_collections

def record_clip_rewatch(chunk_id: str, collection_name: str = "transcript_chunks"):
    """Called when student replays a video clip"""
    col, vis_col = get_or_create_collections()
    target = col if collection_name == "transcript_chunks" else vis_col
    result = target.get(ids=[chunk_id], include=["metadatas"])
    if result["ids"]:
        meta = result["metadatas"][0]
        meta["rewatch_count"] = meta.get("rewatch_count", 0) + 1
        target.update(ids=[chunk_id], metadatas=[meta])

def get_confusion_report(course_id: str, top_n: int = 10) -> list[dict]:
    """Instructor dashboard: most confusing chunks by rewatch + complexity + hits"""
    col, _ = get_or_create_collections()
    results = col.get(
        where={"course_id": {"$eq": course_id}},
        include=["metadatas", "documents"]
    )
    scored = []
    for meta, doc in zip(results["metadatas"], results["documents"]):
        confusion = (
            meta.get("rewatch_count", 0) * 0.4 +
            meta.get("query_hit_count", 0) * 0.3 +
            meta.get("complexity_score", 0) * 0.3
        )
        scored.append({
            "t_start": meta.get("t_start"),
            "t_end": meta.get("t_end"),
            "lecture_id": meta.get("lecture_id"),
            "confusion_score": round(confusion, 3),
            "text_preview": doc[:150]
        })
    scored.sort(key=lambda x: x["confusion_score"], reverse=True)
    return scored[:top_n]
```

---

## FastAPI Backend — main.py

```python
# backend/main.py
import os, json, shutil, uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import UPLOADS_PATH, FRAMES_PATH
from stage1_ingest import extract_audio, extract_keyframes
from stage2_extract import transcribe_audio, classify_frame, extract_frame_features, compute_complexity_score
from stage3_chunk import chunk_transcript, chunk_visual
from stage4_index import index_transcript_chunks, index_visual_chunks, extract_and_store_concepts
from stage5_query import build_routing_object
from stage6_retrieve import retrieve
from stage7_respond import generate_response
from stage8_analytics import get_confusion_report, record_clip_rewatch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded videos as static files so frontend can play them
Path(UPLOADS_PATH).mkdir(parents=True, exist_ok=True)
app.mount("/videos", StaticFiles(directory=UPLOADS_PATH), name="videos")

# In-memory session store (per student)
SESSIONS: dict[str, list] = {}

# ── Ingestion endpoint ────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest_video(
    file: UploadFile = File(...),
    course_id: str = Form(...),
    lecture_id: str = Form(None)
):
    lecture_id = lecture_id or str(uuid.uuid4())[:8]
    video_path = f"{UPLOADS_PATH}/{lecture_id}_{file.filename}"
    audio_path = f"/tmp/{lecture_id}.wav"

    # Save uploaded file
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"[Ingest] Starting pipeline for {file.filename}")

    # Stage 1
    extract_audio(video_path, audio_path)
    keyframes = extract_keyframes(video_path, lecture_id)

    # Stage 2
    transcript_segments = transcribe_audio(audio_path)
    frame_features = []
    for kf in keyframes:
        ctype = classify_frame(kf["frame_path"])
        feats = extract_frame_features(kf["frame_path"], ctype)
        feats["content_type"] = ctype
        feats["complexity_score"] = compute_complexity_score(feats.get("ocr_text", ""), ctype)
        frame_features.append(feats)

    # Stage 3
    t_chunks = chunk_transcript(transcript_segments, course_id, lecture_id)
    v_chunks = chunk_visual(keyframes, frame_features, course_id, lecture_id)

    # Stage 4
    index_transcript_chunks(t_chunks["fine"] + t_chunks["coarse"])
    index_visual_chunks(v_chunks["fine"])
    extract_and_store_concepts(t_chunks["fine"], course_id)

    return {
        "status": "success",
        "lecture_id": lecture_id,
        "video_url": f"/videos/{lecture_id}_{file.filename}",
        "stats": {
            "transcript_segments": len(transcript_segments),
            "keyframes": len(keyframes),
            "fine_transcript_chunks": len(t_chunks["fine"]),
            "visual_chunks": len(v_chunks["fine"])
        }
    }

# ── Query endpoint ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    course_id: str
    student_id: str

@app.post("/query")
async def query(req: QueryRequest):
    history = SESSIONS.get(req.student_id, [])

    routing = build_routing_object(req.query, history)
    clips = retrieve(routing, req.course_id, history)
    response = generate_response(routing, clips, req.query)

    # Update session
    if clips:
        history.append({
            "query": req.query,
            "rewritten": routing.rewritten_query,
            "intent": routing.intent,
            "t_start": clips[0]["t_start"],
            "t_end": clips[0]["t_end"],
            "chunk_id": clips[0].get("chunk_ids", [None])[0]
        })
        SESSIONS[req.student_id] = history[-10:]  # Keep last 10 turns

    return {
        **response,
        "intent": routing.intent,
        "rewritten_query": routing.rewritten_query,
    }

# ── Rewatch tracking ──────────────────────────────────────────────────────────

class RewatchEvent(BaseModel):
    chunk_id: str

@app.post("/rewatch")
async def rewatch(event: RewatchEvent):
    record_clip_rewatch(event.chunk_id)
    return {"status": "recorded"}

# ── Instructor dashboard ──────────────────────────────────────────────────────

@app.get("/analytics/{course_id}")
async def analytics(course_id: str):
    return {"confusion_report": get_confusion_report(course_id)}

# Run with: uvicorn main:app --reload --port 8000
```

---

## React Frontend

```bash
# In WSL2, from the video_rag/ root directory:
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install axios react-player
```

**Replace `frontend/src/App.jsx` with:**

```jsx
// frontend/src/App.jsx
import { useState, useRef } from "react";
import axios from "axios";
import ReactPlayer from "react-player/file";
import "./App.css";

const API = "http://localhost:8000";

// Generate a simple student ID for this session
const STUDENT_ID = "student_" + Math.random().toString(36).slice(2, 8);

export default function App() {
  const [courseId, setCourseId] = useState("CS101");
  const [lectureId, setLectureId] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");

  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState("");
  const [loading, setLoading] = useState(false);

  const playerRef = useRef(null);

  // ── Upload handler ────────────────────────────────────────────────────────

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress("Uploading...");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("course_id", courseId);

    try {
      setUploadProgress("Processing video — transcribing + extracting frames (this takes a few minutes)...");
      const res = await axios.post(`${API}/ingest`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setLectureId(res.data.lecture_id);
      setVideoUrl(`${API}${res.data.video_url}`);
      setUploadProgress(`✅ Ready! ${res.data.stats.transcript_segments} transcript segments, ${res.data.stats.keyframes} keyframes extracted.`);
      setMessages([{
        role: "system",
        text: `Lecture "${file.name}" loaded. Ask me anything about it!`
      }]);
    } catch (err) {
      setUploadProgress("❌ Error: " + (err.response?.data?.detail || err.message));
    } finally {
      setUploading(false);
    }
  };

  // ── Seek to timestamp ─────────────────────────────────────────────────────

  const seekTo = (seconds) => {
    if (playerRef.current) {
      playerRef.current.seekTo(seconds, "seconds");
    }
  };

  // ── Query handler ─────────────────────────────────────────────────────────

  const handleQuery = async () => {
    if (!inputQuery.trim() || !lectureId) return;

    const userMsg = { role: "user", text: inputQuery };
    setMessages((m) => [...m, userMsg]);
    setInputQuery("");
    setLoading(true);

    try {
      const res = await axios.post(`${API}/query`, {
        query: inputQuery,
        course_id: courseId,
        student_id: STUDENT_ID,
      });

      const assistantMsg = {
        role: "assistant",
        text: res.data.answer,
        clips: res.data.clips,
        intent: res.data.intent,
        rewritten: res.data.rewritten_query !== inputQuery ? res.data.rewritten_query : null,
      };
      setMessages((m) => [...m, assistantMsg]);

      // Auto-seek to the top clip
      if (res.data.clips?.length > 0) {
        seekTo(res.data.clips[0].t_start);
      }
    } catch (err) {
      setMessages((m) => [...m, {
        role: "assistant",
        text: "Error: " + (err.response?.data?.detail || err.message),
        clips: []
      }]);
    } finally {
      setLoading(false);
    }
  };

  // ── Format timestamp ──────────────────────────────────────────────────────
  const fmt = (s) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>🎓 LectureRAG</h1>
        <div className="course-input">
          <label>Course ID:</label>
          <input value={courseId} onChange={(e) => setCourseId(e.target.value)} />
        </div>
      </header>

      <div className="main-layout">
        {/* Left: Video + Upload */}
        <div className="video-panel">
          {!videoUrl ? (
            <div className="upload-area">
              <label className="upload-btn">
                {uploading ? "⏳ Processing..." : "📁 Upload Lecture Video"}
                <input type="file" accept="video/*" onChange={handleUpload} disabled={uploading} hidden />
              </label>
              {uploadProgress && <p className="progress-text">{uploadProgress}</p>}
            </div>
          ) : (
            <>
              <ReactPlayer
                ref={playerRef}
                url={videoUrl}
                controls
                width="100%"
                height="auto"
                style={{ borderRadius: "8px", overflow: "hidden" }}
              />
              {uploadProgress && <p className="progress-text">{uploadProgress}</p>}
            </>
          )}
        </div>

        {/* Right: Chat */}
        <div className="chat-panel">
          <div className="messages">
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                {/* Rewrite indicator */}
                {msg.rewritten && (
                  <div className="rewrite-badge">
                    🔄 Interpreted as: "{msg.rewritten}"
                  </div>
                )}

                {/* Intent badge */}
                {msg.intent && (
                  <span className={`intent-badge intent-${msg.intent}`}>
                    {msg.intent}
                  </span>
                )}

                {/* Answer text */}
                <p>{msg.text}</p>

                {/* Timestamp clips — clicking seeks the video */}
                {msg.clips?.map((clip, j) => (
                  <div key={j} className={`clip-card confidence-${clip.confidence.toLowerCase()}`}>
                    <button
                      className="timestamp-btn"
                      onClick={() => seekTo(clip.t_start)}
                      title="Click to jump to this moment"
                    >
                      ▶ {fmt(clip.t_start)} → {fmt(clip.t_end)}
                    </button>
                    <span className={`confidence-tag ${clip.confidence.toLowerCase()}`}>
                      {clip.confidence}
                    </span>
                    {clip.content_type && clip.content_type !== "talking_head" && (
                      <span className="content-type-tag">{clip.content_type}</span>
                    )}
                    {/* See also related concepts */}
                    {clip.see_also?.length > 0 && (
                      <div className="see-also">
                        <span>See also: </span>
                        {clip.see_also.map((sa, k) => (
                          <button key={k} className="see-also-btn"
                            onClick={() => seekTo(sa.t_start)}>
                            {sa.concept} ({fmt(sa.t_start)})
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ))}
            {loading && <div className="message assistant loading">⏳ Thinking...</div>}
          </div>

          {/* Input area */}
          <div className="input-area">
            <input
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleQuery()}
              placeholder={lectureId ? "Ask about the lecture..." : "Upload a video first"}
              disabled={!lectureId || loading}
            />
            <button onClick={handleQuery} disabled={!lectureId || loading}>
              Send
            </button>
          </div>

          {/* Suggested queries */}
          {lectureId && messages.length <= 1 && (
            <div className="suggestions">
              {[
                "Explain backpropagation",
                "Show me the diagram of gradient descent",
                "Quiz me on chain rule",
                "Where in the video is the MLP section?"
              ].map((q) => (
                <button key={q} className="suggestion-pill"
                  onClick={() => { setInputQuery(q); }}>
                  {q}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

**Replace `frontend/src/App.css` with:**

```css
/* frontend/src/App.css */
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Georgia', serif;
  background: #0f1117;
  color: #e8e8e8;
  height: 100vh;
}

.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.header {
  display: flex;
  align-items: center;
  gap: 24px;
  padding: 12px 24px;
  background: #1a1d27;
  border-bottom: 1px solid #2e3148;
}

.header h1 {
  font-size: 1.4rem;
  font-weight: 600;
  color: #a78bfa;
  letter-spacing: -0.5px;
}

.course-input {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.85rem;
  color: #888;
}

.course-input input {
  background: #252836;
  border: 1px solid #3a3d52;
  border-radius: 6px;
  padding: 4px 10px;
  color: #e8e8e8;
  font-size: 0.85rem;
  width: 100px;
}

.main-layout {
  display: flex;
  flex: 1;
  overflow: hidden;
  gap: 0;
}

/* ── Video Panel ─────────────────────────────────────────────────── */
.video-panel {
  flex: 1.2;
  padding: 20px;
  background: #0f1117;
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow-y: auto;
}

.upload-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 300px;
  border: 2px dashed #3a3d52;
  border-radius: 12px;
  gap: 12px;
}

.upload-btn {
  background: #a78bfa;
  color: white;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.95rem;
  transition: background 0.2s;
}

.upload-btn:hover { background: #9061f9; }

.progress-text {
  font-size: 0.82rem;
  color: #888;
  text-align: center;
  padding: 8px;
}

/* ── Chat Panel ──────────────────────────────────────────────────── */
.chat-panel {
  width: 420px;
  min-width: 380px;
  display: flex;
  flex-direction: column;
  background: #1a1d27;
  border-left: 1px solid #2e3148;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.message {
  padding: 12px 14px;
  border-radius: 10px;
  font-size: 0.88rem;
  line-height: 1.55;
  max-width: 100%;
}

.message.user {
  background: #252836;
  color: #c4b5fd;
  align-self: flex-end;
  max-width: 85%;
}

.message.assistant {
  background: #1e2130;
  border: 1px solid #2e3148;
}

.message.system {
  background: #1a2a1a;
  color: #6ee7b7;
  font-size: 0.82rem;
  text-align: center;
  border-radius: 6px;
}

.message.loading { color: #888; font-style: italic; }

.rewrite-badge {
  font-size: 0.75rem;
  color: #f59e0b;
  background: #2a2208;
  padding: 3px 8px;
  border-radius: 4px;
  margin-bottom: 8px;
  display: inline-block;
}

.intent-badge {
  display: inline-block;
  font-size: 0.7rem;
  padding: 2px 8px;
  border-radius: 10px;
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-family: monospace;
}

.intent-concept    { background: #1e3a5f; color: #60a5fa; }
.intent-visual     { background: #1f3a1f; color: #4ade80; }
.intent-temporal   { background: #3a2a1f; color: #fb923c; }
.intent-comparison { background: #3a1f3a; color: #e879f9; }
.intent-generative { background: #3a3a1f; color: #facc15; }
.intent-locator    { background: #1f3a3a; color: #22d3ee; }

/* ── Clip Cards ──────────────────────────────────────────────────── */
.clip-card {
  margin-top: 8px;
  padding: 8px 10px;
  border-radius: 8px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  font-size: 0.8rem;
}

.clip-card.confidence-high { background: #1a2a1a; border: 1px solid #2d5a2d; }
.clip-card.confidence-low  { background: #2a1a1a; border: 1px solid #5a2d2d; }

.timestamp-btn {
  background: #a78bfa;
  color: white;
  border: none;
  border-radius: 5px;
  padding: 4px 10px;
  cursor: pointer;
  font-size: 0.8rem;
  font-family: monospace;
  transition: background 0.15s;
}

.timestamp-btn:hover { background: #9061f9; }

.confidence-tag {
  font-size: 0.7rem;
  padding: 2px 6px;
  border-radius: 4px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.confidence-tag.high { background: #14532d; color: #4ade80; }
.confidence-tag.low  { background: #450a0a; color: #f87171; }

.content-type-tag {
  font-size: 0.7rem;
  padding: 2px 6px;
  border-radius: 4px;
  background: #252836;
  color: #94a3b8;
}

.see-also {
  width: 100%;
  font-size: 0.75rem;
  color: #888;
  margin-top: 4px;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
}

.see-also-btn {
  background: #252836;
  color: #a78bfa;
  border: 1px solid #3a3d52;
  border-radius: 4px;
  padding: 2px 6px;
  cursor: pointer;
  font-size: 0.72rem;
  transition: background 0.15s;
}

.see-also-btn:hover { background: #2e3148; }

/* ── Input Area ──────────────────────────────────────────────────── */
.input-area {
  padding: 12px 16px;
  border-top: 1px solid #2e3148;
  display: flex;
  gap: 8px;
}

.input-area input {
  flex: 1;
  background: #252836;
  border: 1px solid #3a3d52;
  border-radius: 8px;
  padding: 8px 12px;
  color: #e8e8e8;
  font-size: 0.88rem;
  outline: none;
  transition: border-color 0.15s;
}

.input-area input:focus { border-color: #a78bfa; }
.input-area input:disabled { opacity: 0.4; }

.input-area button {
  background: #a78bfa;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 8px 16px;
  cursor: pointer;
  font-size: 0.88rem;
  transition: background 0.15s;
}

.input-area button:hover:not(:disabled) { background: #9061f9; }
.input-area button:disabled { opacity: 0.4; cursor: not-allowed; }

/* ── Suggestions ─────────────────────────────────────────────────── */
.suggestions {
  padding: 8px 16px 12px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.suggestion-pill {
  background: #252836;
  color: #94a3b8;
  border: 1px solid #3a3d52;
  border-radius: 20px;
  padding: 4px 12px;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.15s;
}

.suggestion-pill:hover {
  background: #2e3148;
  color: #c4b5fd;
  border-color: #a78bfa;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3a3d52; border-radius: 3px; }
```

---

## Running Everything

```bash
# Terminal 1 — WSL2: start backend
cd ~/video_rag
source venv/bin/activate
export LLM_PROVIDER=gemini          # or "ollama"
export GEMINI_API_KEY=your_key_here  # from aistudio.google.com
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — WSL2: start frontend
cd ~/video_rag/frontend
npm run dev
# Opens at http://localhost:5173
```

Open `http://localhost:5173` in your Windows browser (WSL2 network is bridged automatically).

---

## Day Schedule

| Time | Task |
|---|---|
| 9:00–9:30 | WSL2 setup: apt installs, pip installs, Ollama install + pull mistral |
| 9:30–10:00 | Download test video (yt-dlp), trim to 30 min (ffmpeg), verify Tesseract works |
| 10:00–11:00 | Stage 1+2: test frame extraction + SSIM, run transcription, inspect output |
| 11:00–12:00 | Stage 3+4: chunking + ChromaDB indexing, verify data stored |
| 12:00–13:00 | LUNCH |
| 13:00–14:00 | Stage 5+6: query routing + retrieval, test 5 queries via curl |
| 14:00–14:30 | Stage 7+8: response gen + analytics |
| 14:30–15:30 | FastAPI main.py, wire up all stages, test /ingest + /query |
| 15:30–17:00 | React frontend: Vite setup, App.jsx + CSS, test full end-to-end in browser |

---

## Quick Test Queries (after ingestion)

```bash
# Concept query
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"explain backpropagation","course_id":"CS101","student_id":"s1"}' | python3 -m json.tool

# Visual query — should route to visual VDB only, filter for diagrams
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"show me the diagram of gradient descent","course_id":"CS101","student_id":"s1"}' | python3 -m json.tool

# Follow-up — tests the query rewriter
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"explain that again more slowly","course_id":"CS101","student_id":"s1"}' | python3 -m json.tool

# Generative
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"quiz me on chain rule","course_id":"CS101","student_id":"s1"}' | python3 -m json.tool
```

---

## Key Design Notes

**No Docker achieved by:**
- ChromaDB replaces Qdrant server → `pip install chromadb`, persists to a folder
- NetworkX replaces Neo4j → pure Python, saves `.graphml` file per course
- Session dict replaces Redis → sufficient for single-user dev
- Tesseract installed via `apt`, not Docker

**SSIM (Structural Similarity Index Measure)** compares brightness, contrast, and local structure simultaneously. It's much better than pixel-diff for slide detection because it doesn't false-trigger on compression artifacts or tiny cursor movements.

**LLM switching:** Set `LLM_PROVIDER=gemini` in your terminal for fast dev. Set `LLM_PROVIDER=ollama` when you want zero API costs. Both go through the single `call_llm()` function in `config.py`.

**Video playback with timestamps:** The React frontend uses `react-player` which wraps the HTML5 video element. Clicking a timestamp badge calls `playerRef.current.seekTo(seconds)` — this is the native browser seek, no server round-trip needed. The video file is served directly from FastAPI's static files mount.
