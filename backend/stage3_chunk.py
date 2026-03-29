import spacy
nlp = spacy.load("en_core_web_sm")

FINE_CHUNK_SENTENCES = 4   # Target sentences per fine chunk

def chunk_transcript(segments: list[dict], course_id: str, lecture_id: str) -> dict:
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
            "linked_visual_chunk_ids": [],
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


def chunk_visual(keyframes: list[dict], features: list[dict], course_id: str, lecture_id: str) -> dict:
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
