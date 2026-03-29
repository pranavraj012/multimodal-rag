# multimodal-rag

## Tech Stack (Verified Versions)
- Python 3.11.4
- FastAPI 0.135.2
- Uvicorn 0.42.0
- ChromaDB 1.5.5
- NetworkX 3.6.1
- spaCy 3.8.14
- sentence-transformers 5.3.0
- faster-whisper 1.2.1
- OpenCV (cv2) 4.13.0
- pytesseract 0.3.13
- google-genai 1.69.0
- Pydantic 2.12.5
- Node.js v24.14.0
- npm 11.9.0
- React 19.2.4
- Vite 8.0.1

## Overview
This project is a multimodal Video RAG system for lecture understanding. It ingests uploaded videos, extracts transcript and visual signals, indexes them, and answers questions with timestamped clips that can be clicked in the UI.

## Pipeline
- Stage 1: Ingest video, extract audio, detect keyframes
- Stage 2: Transcribe audio and extract visual metadata (OCR + sampled VLM)
- Stage 3: Build fine/coarse chunks with timestamps
- Stage 4: Index transcript + visual chunks in ChromaDB, build concept graph
- Stage 5: Query rewrite + intent routing + temporal follow-up handling
- Stage 6: Retrieval, filtering, fusion, re-ranking, graph enrichment
- Stage 7: Answer generation with timestamped clip links, summary/quiz modes
- Stage 8: Analytics and recommendation signals

## Run Locally
### Backend
1. Create and activate venv.
2. Install dependencies from requirements.txt.
3. Start API:

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

### Frontend
```powershell
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:5173 and backend at http://localhost:8000.

## Environment Variables
Create a .env file in project root with optional values:
- LLM_PROVIDER=gemini or ollama
- GEMINI_API_KEY=...
- GEMINI_MODEL=gemini-2.5-flash-lite
- OLLAMA_BASE_URL=http://localhost:11434
- OLLAMA_CHAT_MODEL=llama3.2
- OLLAMA_VISION_MODEL=moondream
- OLLAMA_TIMEOUT_SEC=45
- VISUAL_LLM_MAX_FRAMES=30

## Notes
- `VISUAL_LLM_MAX_FRAMES` caps expensive vision-model calls during ingest for speed.
- The query path is lecture-scoped via `lecture_id`, so retrieval is limited to the active uploaded lecture.
- Timestamp cards in chat are clickable and seek the video player to the retrieved moment.
