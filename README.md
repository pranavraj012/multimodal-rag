# multimodal-rag

Multimodal RAG for lecture videos with timestamp-grounded answers.

## What Is Working Right Now
- Video upload and ingestion pipeline through all 8 stages.
- Audio transcription with faster-whisper.
- Keyframe extraction with OpenCV + SSIM.
- OCR extraction with Tesseract.
- Transcript and visual chunk indexing into ChromaDB.
- Query routing + retrieval scoped by lecture_id.
- Response generation with timestamped clip cards.
- Frontend video player seeking when a timestamp card is clicked.
- Lecture listing endpoint for previously uploaded videos.

## Stack
- Frontend: React + Vite + Axios.
- Backend: FastAPI.
- Retrieval: sentence-transformers + ChromaDB + NetworkX GraphML.
- Vision/OCR: OpenCV + pytesseract.
- Speech: faster-whisper.

## Repository Layout
- backend/ : FastAPI service and pipeline stages.
- frontend/ : React client.
- data/ : Chroma DB, graphs, extracted frames.
- uploads/ : Uploaded videos and extracted audio.

## Prerequisites
- Python 3.10+.
- Node.js 20+.
- FFmpeg installed and available in PATH.
- Tesseract OCR installed.
- Ollama installed and running at http://localhost:11434 if you use local models.
- spaCy English model en_core_web_sm.

Notes:
- You do not need yt-dlp for the current local upload flow.
- You need at least one working LLM path for responses:
  - Ollama running locally, or
  - Gemini API key configured.

## Setup (Windows PowerShell)
1. Clone and open the project.
2. Backend setup:

   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm

3. Frontend setup:

   cd frontend
   npm install
   cd ..

4. Environment file:

   Copy .env.example to .env and edit values.

## Ollama Models In Use Right Now
The backend defaults in code are:
- OLLAMA_CHAT_MODEL=llama3.2
- OLLAMA_VISION_MODEL=moondream

Recommended pulls (explicit latest tags):

ollama pull llama3.2:latest
ollama pull moondream:latest

Then set in .env:

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2:latest
OLLAMA_VISION_MODEL=moondream:latest
OLLAMA_TIMEOUT_SEC=45

Important:
- .env.example currently shows llama3.1:8b, but code defaults use llama3.2.
- If model names in .env do not match what you pulled, requests will fail.

## If You Want To Change Models
1. Pull your preferred chat model and vision model in Ollama.
2. Update only these values in .env:
   - OLLAMA_CHAT_MODEL
   - OLLAMA_VISION_MODEL
3. Restart backend after changing model names.

Examples:
- Lighter/faster chat:

  OLLAMA_CHAT_MODEL=llama3.1:8b

- Keep vision model as moondream unless you install another image-capable model:

  OLLAMA_VISION_MODEL=moondream:latest

## Gemini Mode (Alternative)
You can run generation with Gemini instead of Ollama chat:

LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash-lite

Note:
- Stage 2 visual description currently calls local Ollama for sampled visual frames.
- If Ollama is unavailable, visual descriptions fall back to OCR-based text.

## Run
Backend:

cd backend
uvicorn main:app --reload --port 8000

Frontend:

cd frontend
npm run dev

URLs:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000

## Quick Smoke Test
1. Start backend and frontend.
2. Upload a lecture video in the UI.
3. Ask a question and confirm:
   - Answer text is returned.
   - Clip timestamps are returned.
   - Clicking timestamp seeks video.

## Tests
python -m pytest backend/tests/test_pipeline.py -q

## Runtime Controls
- LLM_PROVIDER: gemini or ollama.
- VISUAL_LLM_BUDGET_MODE: adaptive, fixed, off.
- VISUAL_LLM_MAX_FRAMES: hard cap on visual LLM calls.
- VISUAL_LLM_MIN_FRAMES: adaptive lower floor.
- VISUAL_LLM_FRAME_RATIO: fraction of visual frames to sample.
- VISUAL_LLM_MAX_PER_MIN: duration-based cap per minute.

## Troubleshooting
1. Ollama connection errors:
- Start Ollama service and confirm model is pulled.
- Check OLLAMA_BASE_URL and model names in .env.

2. Slow ingestion:
- Reduce VISUAL_LLM_FRAME_RATIO.
- Use VISUAL_LLM_BUDGET_MODE=fixed and reduce VISUAL_LLM_MAX_FRAMES.

3. spaCy model load errors:
- Run: python -m spacy download en_core_web_sm.

4. OCR not extracting text:
- Confirm Tesseract is installed and available to pytesseract.

5. No answers returned:
- Verify lecture_id is selected/sent from frontend.
- Re-ingest video after major config changes.
