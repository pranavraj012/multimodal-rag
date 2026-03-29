import os, json, shutil, uuid
import time
import math
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import UPLOADS_PATH, FRAMES_PATH
from stage1_ingest import extract_audio, extract_keyframes
from stage2_extract import (
    transcribe_audio,
    classify_frame,
    extract_frame_features,
    compute_complexity_score,
    infer_visual_content_type,
)
from stage3_chunk import chunk_transcript, chunk_visual
from stage4_index import index_transcript_chunks, index_visual_chunks, extract_and_store_concepts
from stage5_query import build_routing_object
from stage6_retrieve import retrieve
from stage7_respond import generate_response
from stage8_analytics import get_confusion_report, get_recommendation_signals, record_clip_rewatch
from config import (
    VISUAL_LLM_BUDGET_MODE,
    VISUAL_LLM_MAX_FRAMES,
    VISUAL_LLM_MIN_FRAMES,
    VISUAL_LLM_FRAME_RATIO,
    VISUAL_LLM_MAX_PER_MIN,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

Path(UPLOADS_PATH).mkdir(parents=True, exist_ok=True)
app.mount("/videos", StaticFiles(directory=UPLOADS_PATH), name="videos")

SESSIONS: dict[str, list] = {}


def _compute_visual_llm_budget(visual_frames: int, duration_sec: float) -> int:
    mode = (VISUAL_LLM_BUDGET_MODE or "adaptive").lower()
    if mode == "off":
        return 0
    if mode == "fixed":
        return max(0, min(visual_frames, VISUAL_LLM_MAX_FRAMES))

    # Adaptive mode: ratio-based budget with duration-aware cap and safety floor.
    ratio_budget = int(round(visual_frames * VISUAL_LLM_FRAME_RATIO))
    duration_minutes = max(duration_sec / 60.0, 1.0)
    duration_budget = int(round(duration_minutes * VISUAL_LLM_MAX_PER_MIN))
    adaptive = min(ratio_budget, duration_budget, VISUAL_LLM_MAX_FRAMES)
    adaptive = max(min(VISUAL_LLM_MIN_FRAMES, VISUAL_LLM_MAX_FRAMES), adaptive)
    return max(0, min(visual_frames, adaptive))

@app.post("/ingest")
async def ingest_video(
    file: UploadFile = File(...),
    course_id: str | None = Form(None),
    lecture_id: str = Form(None)
):
    course_id = (course_id or "general").strip() or "general"
    lecture_id = lecture_id or str(uuid.uuid4())[:8]
    video_path = f"{UPLOADS_PATH}/{lecture_id}_{file.filename}"
    audio_path = f"{UPLOADS_PATH}/{lecture_id}.wav"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"[Ingest] Starting pipeline for {file.filename}")

    t0 = time.perf_counter()
    t_stage1_start = t0

    extract_audio(video_path, audio_path)
    keyframes = extract_keyframes(video_path, lecture_id)
    stage1_seconds = round(time.perf_counter() - t_stage1_start, 2)

    print(f"[Stage 1] Extracted {len(keyframes)} keyframes.")

    print("[Stage 2] Transcribing audio with Faster-Whisper (this may take 1-3 minutes)...")
    t_stage2_transcribe_start = time.perf_counter()
    transcript_segments = transcribe_audio(audio_path)
    stage2_transcribe_seconds = round(time.perf_counter() - t_stage2_transcribe_start, 2)
    
    frame_features = []
    total_kf = len(keyframes)
    classified_types = [classify_frame(kf["frame_path"]) for kf in keyframes]
    visual_indices = [i for i, t in enumerate(classified_types) if t != "talking_head"]
    duration_sec = float(keyframes[-1]["timestamp"]) if keyframes else 0.0

    sampled_visual_indices = set(visual_indices)
    visual_budget = _compute_visual_llm_budget(len(visual_indices), duration_sec)
    if len(visual_indices) > visual_budget > 0:
        sampled_visual_indices = {
            visual_indices[int(j * len(visual_indices) / visual_budget)]
            for j in range(visual_budget)
        }
    elif visual_budget == 0:
        sampled_visual_indices = set()

    estimated_vlm_calls = len(sampled_visual_indices)
    print(f"[Stage 2] Extracting visual features for {total_kf} frames using local VLM...")
    print(
        f"[Stage 2] VLM budgeting mode={VISUAL_LLM_BUDGET_MODE}; "
        f"planned calls={estimated_vlm_calls} (from {len(visual_indices)} visual frames)."
    )
    t_stage2_visual_start = time.perf_counter()
    visual_llm_calls = 0
    
    for i, kf in enumerate(keyframes, 1):
        print(f"  -> Processing frame {i}/{total_kf}...", end="\\r")
        ctype = classified_types[i - 1]
        use_vlm = (i - 1) in sampled_visual_indices
        feats = extract_frame_features(kf["frame_path"], ctype, use_vlm=use_vlm)
        if feats.get("used_vlm"):
            visual_llm_calls += 1
        inferred_type = infer_visual_content_type(
            ctype,
            feats.get("ocr_text", ""),
            feats.get("visual_description", ""),
        )
        feats["content_type"] = inferred_type
        feats["complexity_score"] = compute_complexity_score(feats.get("ocr_text", ""), ctype)
        frame_features.append(feats)
    print("\\n[Stage 2] Visual extraction complete.")
    stage2_visual_seconds = round(time.perf_counter() - t_stage2_visual_start, 2)

    print("[Stage 3] Chunking data...")
    t_stage3_start = time.perf_counter()
    t_chunks = chunk_transcript(transcript_segments, course_id, lecture_id)
    v_chunks = chunk_visual(keyframes, frame_features, course_id, lecture_id)
    stage3_seconds = round(time.perf_counter() - t_stage3_start, 2)

    print("[Stage 4] Indexing to ChromaDB and Knowledge Graph...")
    t_stage4_start = time.perf_counter()
    index_transcript_chunks(t_chunks["fine"] + t_chunks["coarse"])
    index_visual_chunks(v_chunks["fine"])
    extract_and_store_concepts(t_chunks["fine"], course_id)
    stage4_seconds = round(time.perf_counter() - t_stage4_start, 2)
    total_seconds = round(time.perf_counter() - t0, 2)

    return {
        "status": "success",
        "course_id": course_id,
        "lecture_id": lecture_id,
        "video_url": f"/videos/{lecture_id}_{file.filename}",
        "stats": {
            "transcript_segments": len(transcript_segments),
            "keyframes": len(keyframes),
            "fine_transcript_chunks": len(t_chunks["fine"]),
            "visual_chunks": len(v_chunks["fine"]),
            "visual_frames": len(visual_indices),
            "visual_llm_calls": visual_llm_calls,
            "visual_llm_budget": visual_budget,
            "visual_llm_budget_mode": VISUAL_LLM_BUDGET_MODE,
            "timings": {
                "stage1_ingest_seconds": stage1_seconds,
                "stage2_transcribe_seconds": stage2_transcribe_seconds,
                "stage2_visual_seconds": stage2_visual_seconds,
                "stage3_chunk_seconds": stage3_seconds,
                "stage4_index_seconds": stage4_seconds,
                "total_seconds": total_seconds,
            }
        }
    }

class QueryRequest(BaseModel):
    query: str
    course_id: str | None = None
    lecture_id: str
    student_id: str

@app.post("/query")
async def query(req: QueryRequest):
    history = SESSIONS.get(req.student_id, [])

    routing = build_routing_object(req.query, history)
    clips = retrieve(routing, req.course_id, history, lecture_id=req.lecture_id)
    response = generate_response(routing, clips, req.query)

    if clips:
        history.append({
            "query": req.query,
            "rewritten": routing.rewritten_query,
            "intent": routing.intent,
            "t_start": clips[0]["t_start"],
            "t_end": clips[0]["t_end"],
            "chunk_id": clips[0].get("chunk_ids", [None])[0]
        })
        SESSIONS[req.student_id] = history[-10:]

    return {
        **response,
        "intent": routing.intent,
        "rewritten_query": routing.rewritten_query,
    }

class RewatchEvent(BaseModel):
    chunk_id: str

@app.post("/rewatch")
async def rewatch(event: RewatchEvent):
    record_clip_rewatch(event.chunk_id)
    return {"status": "recorded"}

@app.get("/analytics/{course_id}")
async def analytics(course_id: str):
    return {
        "confusion_report": get_confusion_report(course_id),
        "recommendation_signals": get_recommendation_signals(course_id),
    }
