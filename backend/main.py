import os, json, shutil, uuid
import time
import math
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import UPLOADS_PATH, FRAMES_PATH
from stage1_ingest import extract_audio, extract_keyframes
from stage2_extract import (
    transcribe_audio,
    analyze_frame_quick,
    extract_frame_features,
    compute_complexity_score,
    infer_visual_content_type,
)
from stage3_chunk import chunk_transcript, chunk_visual
from stage4_index import (
    ensure_embedding_backend_ready,
    get_or_create_collections,
    get_chroma,
    index_transcript_chunks,
    index_visual_chunks,
    extract_and_store_concepts,
)
from stage5_query import build_routing_object
from stage6_retrieve import retrieve
from stage7_respond import generate_response
from stage8_analytics import get_confusion_report, get_recommendation_signals, record_clip_rewatch
from config import (
    GRAPHS_PATH,
    VISUAL_LLM_BUDGET_MODE,
    VISUAL_LLM_MAX_FRAMES,
    VISUAL_LLM_MIN_FRAMES,
    VISUAL_LLM_FRAME_RATIO,
    VISUAL_LLM_MAX_PER_MIN,
    OCR_EVERY_N_VISUAL_FRAMES,
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
    current_stage = "preflight"

    t0 = time.perf_counter()
    try:
        print("[Ingest] Preflight: checking embedding backend/model availability...")
        ensure_embedding_backend_ready()
        print("[Ingest] Preflight OK")

        current_stage = "stage1"
        t_stage1_start = t0

        extract_audio(video_path, audio_path)
        keyframes = extract_keyframes(video_path, lecture_id)
        stage1_seconds = round(time.perf_counter() - t_stage1_start, 2)
        print(f"[Stage 1] Extracted {len(keyframes)} keyframes in {stage1_seconds}s")

        current_stage = "stage2"
        print("[Stage 2] Transcribing audio with Faster-Whisper (this may take 1-3 minutes)...")
        t_stage2_transcribe_start = time.perf_counter()
        transcript_segments = transcribe_audio(audio_path)
        stage2_transcribe_seconds = round(time.perf_counter() - t_stage2_transcribe_start, 2)
        print(f"[Stage 2] Transcription done in {stage2_transcribe_seconds}s")

        frame_features = []
        total_kf = len(keyframes)
        t_stage2_classify_start = time.perf_counter()
        quick_analysis = [analyze_frame_quick(kf["frame_path"]) for kf in keyframes]
        classified_types = [x[0] for x in quick_analysis]
        priority_scores = [x[1] for x in quick_analysis]
        stage2_classify_seconds = round(time.perf_counter() - t_stage2_classify_start, 2)
        visual_indices = [i for i, t in enumerate(classified_types) if t != "talking_head"]
        talking_head_count = total_kf - len(visual_indices)
        duration_sec = float(keyframes[-1]["timestamp"]) if keyframes else 0.0
        print(f"[Stage 2] Frame classification done in {stage2_classify_seconds}s")

        t_stage2_sampling_start = time.perf_counter()
        sampled_visual_indices = set(visual_indices)
        visual_budget = _compute_visual_llm_budget(len(visual_indices), duration_sec)
        if len(visual_indices) > visual_budget > 0:
            scored = []
            for vidx in visual_indices:
                score = priority_scores[vidx]
                scored.append((score, vidx))
            scored.sort(key=lambda x: x[0], reverse=True)

            priority_quota = max(1, int(round(visual_budget * 0.7)))
            priority_quota = min(priority_quota, visual_budget)
            coverage_quota = max(0, visual_budget - priority_quota)

            top_priority = [idx for _, idx in scored[:priority_quota]]
            coverage = []
            if coverage_quota > 0:
                for j in range(coverage_quota):
                    pick = visual_indices[int(j * len(visual_indices) / max(coverage_quota, 1))]
                    coverage.append(pick)

            sampled_visual_indices = set(top_priority + coverage)

            # If dedup caused underfill, backfill from next best scored frames.
            if len(sampled_visual_indices) < visual_budget:
                for _, vidx in scored[priority_quota:]:
                    sampled_visual_indices.add(vidx)
                    if len(sampled_visual_indices) >= visual_budget:
                        break
        elif visual_budget == 0:
            sampled_visual_indices = set()
        stage2_sampling_seconds = round(time.perf_counter() - t_stage2_sampling_start, 2)

        estimated_vlm_calls = len(sampled_visual_indices)
        print(
            f"[Stage 2] Keyframes={total_kf} | visual={len(visual_indices)} | "
            f"talking_head={talking_head_count}"
        )
        print(f"[Stage 2] Extracting visual features for {total_kf} keyframes...")
        print(
            f"[Stage 2] VLM budgeting mode={VISUAL_LLM_BUDGET_MODE}; "
            f"planned calls={estimated_vlm_calls} (from {len(visual_indices)} visual frames)."
        )
        print(f"[Stage 2] Frame sampling done in {stage2_sampling_seconds}s")
        print(f"[Stage 2] OCR stride on non-VLM visual frames: every {max(1, OCR_EVERY_N_VISUAL_FRAMES)} frame(s)")
        if estimated_vlm_calls > 0:
            print("[Stage 2] VLM frame policy: 70% high-priority + 30% timeline coverage")
        t_stage2_visual_start = time.perf_counter()
        visual_llm_calls = 0
        visual_seen = 0

        progress_every = max(25, total_kf // 50) if total_kf else 25
        for i, kf in enumerate(keyframes, 1):
            if i == 1 or i == total_kf or i % progress_every == 0:
                print(f"[Stage 2] Processing frame {i}/{total_kf}...")
            ctype = classified_types[i - 1]
            use_vlm = (i - 1) in sampled_visual_indices
            use_ocr = True
            if ctype != "talking_head":
                visual_seen += 1
                if (not use_vlm) and max(1, OCR_EVERY_N_VISUAL_FRAMES) > 1:
                    use_ocr = (visual_seen % max(1, OCR_EVERY_N_VISUAL_FRAMES) == 0)

            feats = extract_frame_features(kf["frame_path"], ctype, use_vlm=use_vlm, use_ocr=use_ocr)
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
        stage2_visual_seconds = round(time.perf_counter() - t_stage2_visual_start, 2)
        stage2_total_seconds = round(
            stage2_transcribe_seconds + stage2_classify_seconds + stage2_sampling_seconds + stage2_visual_seconds,
            2,
        )
        print(f"[Stage 2] Visual extraction complete in {stage2_visual_seconds}s")
        print(
            f"[Stage 2] Total stage time: {stage2_total_seconds}s "
            f"(transcribe={stage2_transcribe_seconds}s, classify={stage2_classify_seconds}s, "
            f"sample={stage2_sampling_seconds}s, visual={stage2_visual_seconds}s)"
        )

        current_stage = "stage3"
        print("[Stage 3] Chunking data...")
        t_stage3_start = time.perf_counter()
        t_chunks = chunk_transcript(transcript_segments, course_id, lecture_id)
        v_chunks = chunk_visual(keyframes, frame_features, course_id, lecture_id)
        stage3_seconds = round(time.perf_counter() - t_stage3_start, 2)
        print(f"[Stage 3] Completed in {stage3_seconds}s")

        current_stage = "stage4"
        print("[Stage 4] Indexing to ChromaDB and Knowledge Graph...")
        t_stage4_start = time.perf_counter()
        t_stage4a = time.perf_counter()
        index_transcript_chunks(t_chunks["fine"] + t_chunks["coarse"])
        stage4_transcript_seconds = round(time.perf_counter() - t_stage4a, 2)

        t_stage4b = time.perf_counter()
        index_visual_chunks(v_chunks["fine"])
        stage4_visual_seconds = round(time.perf_counter() - t_stage4b, 2)

        t_stage4c = time.perf_counter()
        extract_and_store_concepts(t_chunks["fine"], course_id)
        stage4_graph_seconds = round(time.perf_counter() - t_stage4c, 2)

        stage4_seconds = round(time.perf_counter() - t_stage4_start, 2)
        print(
            f"[Stage 4] Completed in {stage4_seconds}s "
            f"(transcript={stage4_transcript_seconds}s, visual={stage4_visual_seconds}s, graph={stage4_graph_seconds}s)"
        )
        total_seconds = round(time.perf_counter() - t0, 2)
        print(
            f"[Ingest] Done in {total_seconds}s "
            f"(S1={stage1_seconds}s, S2={stage2_total_seconds}s, S3={stage3_seconds}s, S4={stage4_seconds}s)"
        )
    except Exception as e:
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"[Ingest] Failed at {current_stage} after {elapsed}s: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed at {current_stage}: {e}")

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
                "stage2_classify_seconds": stage2_classify_seconds,
                "stage2_sampling_seconds": stage2_sampling_seconds,
                "stage2_visual_seconds": stage2_visual_seconds,
                "stage2_total_seconds": stage2_total_seconds,
                "stage3_chunk_seconds": stage3_seconds,
                "stage4_index_seconds": stage4_seconds,
                "stage4_transcript_seconds": stage4_transcript_seconds,
                "stage4_visual_seconds": stage4_visual_seconds,
                "stage4_graph_seconds": stage4_graph_seconds,
                "total_seconds": total_seconds,
            }
        }
    }

class QueryRequest(BaseModel):
    query: str
    course_id: str | None = None
    lecture_id: str
    student_id: str
    use_rerank: bool = True

@app.post("/query")
async def query(req: QueryRequest):
    history = SESSIONS.get(req.student_id, [])

    routing = build_routing_object(req.query, history)
    clips = retrieve(
        routing,
        req.course_id,
        history,
        lecture_id=req.lecture_id,
        use_rerank=req.use_rerank,
    )
    response = generate_response(routing, clips, req.query)

    if clips:
        first_clip = clips[0]
        t_start = first_clip.get("t_start")
        t_end = first_clip.get("t_end")
        if t_start is None or t_end is None:
            t_start = 0.0
            t_end = 0.0
        history.append({
            "query": req.query,
            "rewritten": routing.rewritten_query,
            "intent": routing.intent,
            "t_start": float(t_start),
            "t_end": float(t_end),
            "chunk_id": first_clip.get("chunk_ids", [None])[0]
        })
        SESSIONS[req.student_id] = history[-10:]

    return {
        **response,
        "intent": routing.intent,
        "rewritten_query": routing.rewritten_query,
        "retrieval_context": [
            (c.get("context_text") or "").strip()
            for c in clips
            if (c.get("context_text") or "").strip()
        ],
    }

class RewatchEvent(BaseModel):
    chunk_id: str

@app.post("/rewatch")
async def rewatch(event: RewatchEvent):
    record_clip_rewatch(event.chunk_id)
    return {"status": "recorded"}


class ClearSessionRequest(BaseModel):
    student_id: str


class ClearLecturesRequest(BaseModel):
    remove_graphs: bool = True


@app.post("/session/clear")
async def clear_session(req: ClearSessionRequest):
    SESSIONS.pop(req.student_id, None)
    return {"status": "cleared"}


@app.post("/lectures/clear")
async def clear_lectures(req: ClearLecturesRequest):
    uploads_dir = Path(UPLOADS_PATH)
    frames_dir = Path(FRAMES_PATH)

    deleted_upload_files = 0
    deleted_frame_dirs = 0

    if uploads_dir.exists():
        for f in uploads_dir.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                    deleted_upload_files += 1
                except Exception:
                    pass

    if frames_dir.exists():
        for d in frames_dir.iterdir():
            if d.is_dir():
                try:
                    shutil.rmtree(d, ignore_errors=True)
                    deleted_frame_dirs += 1
                except Exception:
                    pass

    try:
        client = get_chroma()
        for name in ["transcript_chunks", "visual_chunks"]:
            try:
                client.delete_collection(name=name)
            except Exception:
                pass
    except Exception:
        pass

    deleted_graphs = 0
    if req.remove_graphs:
        graphs_dir = Path(GRAPHS_PATH)
        if graphs_dir.exists():
            for g in graphs_dir.glob("*.graphml"):
                try:
                    g.unlink()
                    deleted_graphs += 1
                except Exception:
                    pass

    return {
        "status": "cleared",
        "deleted_upload_files": deleted_upload_files,
        "deleted_frame_dirs": deleted_frame_dirs,
        "deleted_graphs": deleted_graphs,
    }

@app.get("/analytics/{course_id}")
async def analytics(course_id: str):
    return {
        "confusion_report": get_confusion_report(course_id),
        "recommendation_signals": get_recommendation_signals(course_id),
    }


@app.get("/lectures")
async def list_lectures(course_id: str | None = Query(default=None)):
    uploads_dir = Path(UPLOADS_PATH)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Build metadata lookup from indexed transcript chunks.
    lecture_meta = {}
    try:
        transcript_col, _ = get_or_create_collections()
        data = transcript_col.get(include=["metadatas"])
        for meta in data.get("metadatas", []) or []:
            lid = meta.get("lecture_id")
            if not lid:
                continue
            if lid not in lecture_meta:
                lecture_meta[lid] = {
                    "course_id": meta.get("course_id", "general"),
                }
    except Exception:
        # If index cannot be read, we still list uploaded videos.
        pass

    lectures = {}
    for f in uploads_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in {".wav", ".mp3", ".aac", ".m4a"}:
            continue
        if "_" not in f.name:
            continue

        lecture_id, original_name = f.name.split("_", 1)
        meta = lecture_meta.get(lecture_id, {"course_id": "general"})

        if course_id and meta.get("course_id") != course_id:
            continue

        existing = lectures.get(lecture_id)
        record = {
            "lecture_id": lecture_id,
            "course_id": meta.get("course_id", "general"),
            "file_name": original_name,
            "video_url": f"/videos/{f.name}",
            "updated_at": f.stat().st_mtime,
        }
        if existing is None or record["updated_at"] > existing["updated_at"]:
            lectures[lecture_id] = record

    items = sorted(lectures.values(), key=lambda x: x["updated_at"], reverse=True)
    return {
        "lectures": [
            {
                "lecture_id": x["lecture_id"],
                "course_id": x["course_id"],
                "file_name": x["file_name"],
                "video_url": x["video_url"],
            }
            for x in items
        ]
    }
