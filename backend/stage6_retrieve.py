import json
import networkx as nx
from pathlib import Path
from stage5_query import RoutingObject
from stage4_index import get_or_create_collections, get_embedder, load_graph
from config import call_llm
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def retrieve(
    routing: RoutingObject,
    course_id: str | None,
    session_history: list,
    top_k: int = 5,
    lecture_id: str | None = None,
    use_rerank: bool = True,
) -> list[dict]:
    embedder = get_embedder()
    query_vector = embedder.encode(routing.rewritten_query).tolist()
    transcript_col, visual_col = get_or_create_collections()

    raw_hits = []

    def _with_scope(base_conditions: list[dict]) -> dict:
        scoped = list(base_conditions)
        if lecture_id:
            scoped.append({"lecture_id": {"$eq": lecture_id}})
        if course_id:
            scoped.append({"course_id": {"$eq": course_id}})
        if not scoped:
            return {}
        return {"$and": scoped}

    if "transcript" in routing.search_targets:
        fine_where = _with_scope([
            {"chunk_type": {"$eq": "fine"}}
        ])
        fine_results = transcript_col.query(
            query_embeddings=[query_vector],
            n_results=min(top_k * 3, 30),
            where=fine_where,
            include=["metadatas", "documents", "distances"]
        )
        raw_hits.extend(_results_to_hits(fine_results, "transcript"))

        # Coarse fallback for broad/generative prompts that may not align strongly with fine chunks.
        if routing.granularity == "coarse_then_fine":
            coarse_where = _with_scope([
                {"chunk_type": {"$eq": "coarse"}}
            ])
            coarse_results = transcript_col.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, 5),
                where=coarse_where,
                include=["metadatas", "documents", "distances"]
            )
            raw_hits.extend(_results_to_hits(coarse_results, "transcript"))

    if "visual" in routing.search_targets:
        vis_where_conditions = []
        if routing.content_type_filter:
            vis_where_conditions.append({"content_type": {"$eq": routing.content_type_filter}})
        vis_where = _with_scope(vis_where_conditions)

        results = visual_col.query(
            query_embeddings=[query_vector],
            n_results=min(top_k * 2, 20),
            where=vis_where,
            include=["metadatas", "documents", "distances"]
        )
        visual_hits = _results_to_hits(results, "visual")

        # Fallback: if a strict content filter (e.g., diagram) yields nothing, retry without it.
        if not visual_hits and routing.content_type_filter:
            broad_results = visual_col.query(
                query_embeddings=[query_vector],
                n_results=min(top_k * 3, 30),
                where=_with_scope([]),
                include=["metadatas", "documents", "distances"]
            )
            visual_hits = _results_to_hits(broad_results, "visual")

        raw_hits.extend(visual_hits)

    if routing.temporal_anchor:
        raw_hits = _apply_temporal_anchor(raw_hits, routing)

    print(
        f"[Stage 6] intent={routing.intent} | targets={routing.search_targets} | "
        f"raw_hits={len(raw_hits)}"
    )

    seen_ids = {h.get("chunk_id") for h in session_history}
    min_score = _intent_min_score(routing.intent)
    filtered = [h for h in raw_hits if h["chunk_id"] not in seen_ids and h["score"] >= min_score]

    # If the threshold is too strict for this lecture/query, keep best unseen hits instead of failing hard.
    if not filtered:
        unseen = [h for h in raw_hits if h["chunk_id"] not in seen_ids]
        filtered = sorted(unseen, key=lambda x: x["score"], reverse=True)[: max(top_k, 3)]

    if not filtered and routing.intent == "visual":
        filtered = _fallback_visual_by_complexity(visual_col, course_id, top_k, lecture_id=lecture_id)

    # Infer video duration from the furthest timestamp seen in hits to scale offsets dynamically.
    video_duration = _infer_video_duration(raw_hits)
    fusion_gap, context_window = _adaptive_offsets(video_duration)
    print(f"[Stage 6] video_duration≈{video_duration:.0f}s | fusion_gap={fusion_gap}s | context_window={context_window}s")

    clips = _fuse_into_clips(filtered, gap_threshold=fusion_gap)

    if routing.intent == "visual" and clips:
        _attach_transcript_context_to_clips(clips, transcript_col, course_id, lecture_id=lecture_id, window_sec=context_window)

    for clip in clips:
        if use_rerank:
            popularity = min(clip["query_hit_count"] / 50.0, 1.0)
            recency_bonus = 0.1 if clip["t_start"] < 300 else 0.0
            temporal_bonus = _temporal_bonus(clip, routing.temporal_anchor)
            base_score = (
                0.5 * clip["clip_score"] +
                0.3 * popularity +
                0.15 * recency_bonus +
                0.05 * temporal_bonus
            )
            clip["final_score"] = base_score + _heuristic_rerank_bonus(clip, routing)
        else:
            clip["final_score"] = clip.get("clip_score", 0.0)

    clips.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    if clips:
        top = clips[0]
        print(
            f"[Stage 6] fused_clips={len(clips)} | top={top['t_start']:.1f}s-{top['t_end']:.1f}s "
            f"score={top.get('final_score', 0.0):.3f} conf={top.get('confidence', 'LOW')}"
        )

    if routing.intent in ("concept", "comparison") and clips:
        clips = _enrich_with_graph(clips, routing.rewritten_query, course_id)

    # Apply LLM Text-based Reranking to the top subset
    clips = clips[:top_k * 2]
    if use_rerank and clips and routing.intent not in ("generative",):
        clips = _llm_rerank_clips(clips, routing.rewritten_query)

    _bump_query_hit_counts(clips)

    return clips[:top_k]


def _infer_video_duration(hits: list[dict]) -> float:
    """Estimate total video length from the largest t_end seen in all raw hits."""
    max_t = 0.0
    for h in hits:
        t = float(h.get("meta", {}).get("t_end", 0.0))
        if t > max_t:
            max_t = t
    return max_t if max_t > 0 else 600.0  # default to 10 min if unknown


def _adaptive_offsets(duration: float) -> tuple[float, float]:
    """Return (fusion_gap_sec, context_window_sec) scaled to video length."""
    if duration < 60:          # < 1 minute
        return 10.0, 8.0
    elif duration < 300:       # 1–5 minutes
        return 18.0, 15.0
    elif duration < 900:       # 5–15 minutes
        return 25.0, 20.0
    else:                      # 15+ minutes (lectures)
        return 30.0, 25.0


def _results_to_hits(results: dict, source: str) -> list[dict]:
    hits = []
    if not results or not results.get("metadatas") or not results["metadatas"]:
        return hits

    for i, (meta, doc, dist) in enumerate(zip(
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0]
    )):
        score = max(0.0, 1 - float(dist))
        hits.append({
            "source": source,
            "rank": i,
            "score": score,
            "meta": meta,
            "text": doc,
            "chunk_id": results["ids"][0][i],
            "chunk_type": meta.get("chunk_type", "fine"),
        })
    return hits


def _intent_min_score(intent: str) -> float:
    thresholds = {
        "generative": 0.0,
        "concept": 0.12,
        "comparison": 0.12,
        "visual": 0.18,
        "temporal": 0.05,
        "locator": 0.05,
    }
    return thresholds.get(intent, 0.12)


def _apply_temporal_anchor(hits: list[dict], routing: RoutingObject) -> list[dict]:
    anchor = routing.temporal_anchor or {}
    if "t_start" not in anchor or "t_end" not in anchor:
        return hits

    try:
        if anchor.get("t_start") is None or anchor.get("t_end") is None:
            return hits
        a_start = float(anchor["t_start"])
        a_end = float(anchor["t_end"])
    except (TypeError, ValueError):
        return hits
    center = (a_start + a_end) / 2.0
    window = 120.0
    intent = (routing.intent or "").lower()
    q = (routing.rewritten_query or "").lower()

    filtered = []
    for h in hits:
        t_start = float(h["meta"].get("t_start", 0.0))
        t_end = float(h["meta"].get("t_end", t_start + 10.0))
        h_center = (t_start + t_end) / 2.0

        if intent == "temporal" and ("before" in q or "earlier" in q or "previous" in q):
            if h_end_before_anchor(t_end, a_start) and abs(h_center - center) <= window:
                h["score"] = min(1.0, h["score"] + 0.12)
                filtered.append(h)
            continue

        if intent == "temporal" and ("after" in q or "later" in q or "next" in q):
            if h_start_after_anchor(t_start, a_end) and abs(h_center - center) <= window:
                h["score"] = min(1.0, h["score"] + 0.12)
                filtered.append(h)
            continue

        # Default: keep nearby clips for follow-up references.
        if abs(h_center - center) <= window:
            h["score"] = min(1.0, h["score"] + 0.08)
            filtered.append(h)

    return filtered if filtered else hits


def h_end_before_anchor(t_end: float, a_start: float) -> bool:
    return t_end <= a_start


def h_start_after_anchor(t_start: float, a_end: float) -> bool:
    return t_start >= a_end


def _temporal_bonus(clip: dict, temporal_anchor: dict | None) -> float:
    if not temporal_anchor:
        return 0.0
    if "t_start" not in temporal_anchor or "t_end" not in temporal_anchor:
        return 0.0
    if temporal_anchor.get("t_start") is None or temporal_anchor.get("t_end") is None:
        return 0.0
    try:
        a_start = float(temporal_anchor["t_start"])
        a_end = float(temporal_anchor["t_end"])
    except (TypeError, ValueError):
        return 0.0
    c_center = (float(clip.get("t_start", 0.0)) + float(clip.get("t_end", 0.0))) / 2.0
    a_center = (a_start + a_end) / 2.0
    dist = abs(c_center - a_center)
    return max(0.0, 1.0 - (dist / 180.0))


def _heuristic_rerank_bonus(clip: dict, routing: RoutingObject) -> float:
    bonus = 0.0

    modalities = clip.get("modalities", set())
    content_type = (clip.get("content_type") or "").lower()
    duration = max(0.0, float(clip.get("t_end", 0.0)) - float(clip.get("t_start", 0.0)))

    if len(modalities) >= 2:
        bonus += 0.08

    if routing.intent == "visual":
        if content_type in {"diagram", "code"}:
            bonus += 0.07
        if "visual" in modalities:
            bonus += 0.04

    # Prefer tighter clips; overly broad clips are usually less actionable.
    if duration > 120:
        bonus -= 0.06
    elif duration <= 45:
        bonus += 0.03

    return bonus


def _bump_query_hit_counts(clips: list[dict]):
    if not clips:
        return

    transcript_col, visual_col = get_or_create_collections()
    all_ids = []
    for clip in clips:
        all_ids.extend(clip.get("chunk_ids", []))

    # De-duplicate while preserving order.
    unique_ids = list(dict.fromkeys([cid for cid in all_ids if cid]))
    if not unique_ids:
        return

    t_ids = [cid for cid in unique_ids if "_t_" in cid]
    v_ids = [cid for cid in unique_ids if "_v_" in cid]

    if t_ids:
        _increment_metadata_counter(transcript_col, t_ids, "query_hit_count")
    if v_ids:
        _increment_metadata_counter(visual_col, v_ids, "query_hit_count")


def _increment_metadata_counter(collection, ids: list[str], counter_key: str):
    data = collection.get(ids=ids, include=["metadatas"])
    if not data or not data.get("ids"):
        return

    updates = []
    for meta in data.get("metadatas", []):
        m = dict(meta)
        m[counter_key] = int(m.get(counter_key, 0)) + 1
        updates.append(m)

    if updates:
        collection.update(ids=data["ids"], metadatas=updates)


def _attach_transcript_context_to_clips(
    clips: list[dict],
    transcript_col,
    course_id: str | None,
    lecture_id: str | None = None,
    window_sec: float = 25.0,
):
    for clip in clips:
        existing = (clip.get("context_text") or "").strip()
        if len(existing) >= 120:
            continue

        t_start = float(clip.get("t_start", 0.0))
        t_end = float(clip.get("t_end", t_start + 10.0))
        where_conditions = [
            {"chunk_type": {"$eq": "fine"}},
            {"t_start": {"$lte": t_end + window_sec}},
            {"t_end": {"$gte": t_start - window_sec}},
        ]
        if lecture_id:
            where_conditions.append({"lecture_id": {"$eq": lecture_id}})
        if course_id:
            where_conditions.append({"course_id": {"$eq": course_id}})

        results = transcript_col.get(
            where={"$and": where_conditions},
            include=["metadatas", "documents"],
        )

        if not results or not results.get("documents"):
            continue

        pairs = list(zip(results["metadatas"], results["documents"]))
        pairs.sort(key=lambda x: abs((x[0].get("t_start", 0.0) + x[0].get("t_end", 0.0)) / 2 - (t_start + t_end) / 2))
        transcript_context = " ".join(doc.strip() for _, doc in pairs[:3] if doc).strip()
        if transcript_context:
            if existing:
                clip["context_text"] = f"{existing} {transcript_context}".strip()
            else:
                clip["context_text"] = transcript_context


def _fallback_visual_by_complexity(
    visual_col,
    course_id: str | None,
    top_k: int,
    lecture_id: str | None = None,
) -> list[dict]:
    where_conditions = []
    if lecture_id:
        where_conditions.append({"lecture_id": {"$eq": lecture_id}})
    if course_id:
        where_conditions.append({"course_id": {"$eq": course_id}})

    where_clause = {"$and": where_conditions} if where_conditions else None
    if where_clause:
        data = visual_col.get(
            where=where_clause,
            include=["metadatas", "documents"],
        )
    else:
        data = visual_col.get(include=["metadatas", "documents"])
    if not data or not data.get("ids"):
        return []

    hits = []
    for i, cid in enumerate(data["ids"]):
        meta = data["metadatas"][i]
        doc = data["documents"][i]
        complexity = float(meta.get("complexity_score", 0.0))
        popularity = min(float(meta.get("query_hit_count", 0)) / 50.0, 1.0)
        score = 0.7 * complexity + 0.3 * popularity
        hits.append({
            "source": "visual",
            "rank": i,
            "score": score,
            "meta": meta,
            "text": doc,
            "chunk_id": cid,
        })

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[: max(top_k, 3)]


def _fuse_into_clips(hits: list[dict], gap_threshold: float = 30.0) -> list[dict]:
    clips = []
    for hit in sorted(hits, key=lambda x: x.get("score", 0), reverse=True):
        t_start = hit["meta"].get("t_start", 0)
        t_end = hit["meta"].get("t_end", t_start + 10)
        placed = False
        for clip in clips:
            if abs(t_start - clip["t_start"]) < gap_threshold:
                clip["t_start"] = min(clip["t_start"], t_start)
                clip["t_end"] = max(clip["t_end"], t_end)
                clip["modalities"].add(hit["source"])
                if hit["source"] == "transcript":
                    clip["clip_score"] = max(clip["clip_score"], 0.6 * hit["score"])
                else:
                    clip["clip_score"] = max(clip["clip_score"], 0.4 * hit["score"])
                clip["query_hit_count"] = max(
                    clip["query_hit_count"],
                    hit["meta"].get("query_hit_count", 0)
                )
                clip["context_text"] = clip.get("context_text", "") + " " + hit["text"]
                if hit["chunk_id"] not in clip["chunk_ids"]:
                    clip["chunk_ids"].append(hit["chunk_id"])
                clip["chunk_types"].add(hit.get("chunk_type", "fine"))
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
                "chunk_types": {hit.get("chunk_type", "fine")},
                "lecture_id": hit["meta"].get("lecture_id", ""),
                "content_type": hit["meta"].get("content_type", ""),
            })

    for clip in clips:
        # High confidence when evidence is multimodal OR retrieval score is strong.
        is_multimodal = len(clip["modalities"]) >= 2
        strong_single_modality = float(clip.get("clip_score", 0.0)) >= 0.45
        clip["confidence"] = "HIGH" if (is_multimodal or strong_single_modality) else "LOW"

    has_fine_clip = any("fine" in c.get("chunk_types", set()) for c in clips)
    if has_fine_clip:
        clips = [c for c in clips if "fine" in c.get("chunk_types", set())]

    for clip in clips:
        if "chunk_types" in clip:
            del clip["chunk_types"]

    return clips


def _llm_rerank_clips(clips: list[dict], query: str) -> list[dict]:
    if not clips:
        return clips

    prompt = f"Evaluate the relevance of the following {len(clips)} extracted video clips to the user query: '{query}'.\n"
    prompt += "Assign a score from 0 to 10 for each clip, where 10 is perfectly relevant and 0 is completely irrelevant.\n"
    prompt += "Return ONLY a valid JSON array of numbers, e.g. [8, 0, 5]. Do not include any other text.\n\n"

    for i, clip in enumerate(clips):
        text = (clip.get("context_text") or "")[:400]
        prompt += f"--- Clip {i} ---\n{text}\n\n"

    try:
        response = call_llm(prompt).strip()
        json_match = re.search(r"\[[\d\s,.]+\]", response)
        if json_match:
            scores = json.loads(json_match.group(0))
            if len(scores) == len(clips):
                for i, clip in enumerate(clips):
                    llm_score = min(float(scores[i]) / 10.0, 1.0)
                    clip["final_score"] = float(clip.get("final_score", 0.0)) + (llm_score * 0.8)
                    
                    # Stricter confidence ratings
                    if llm_score >= 0.75:
                        clip["confidence"] = "HIGH"
                    else:
                        clip["confidence"] = "LOW"
                        
                    clip["_llm_drop"] = (llm_score <= 0.2)
            else:
                print(f"[Stage 6] Rerank skip: Mismatched array length {len(scores)} vs {len(clips)}")
        else:
            print("[Stage 6] Could not parse LLM rerank JSON:", response)
    except Exception as e:
        print(f"[Stage 6] LLM Reranking failed: {e}")

    # Drop completely irrelevant clips and sort
    clips = [c for c in clips if not c.get("_llm_drop", False)]
    clips.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    
    # Dynamic Cutoff: Don't show 5 clips. Cap at 3, and drop if score significantly falls off.
    if clips:
        best = clips[0].get("final_score", 0)
        clips = [c for c in clips if c.get("final_score", 0) >= best * 0.6]
        
    return clips[:3]


def _enrich_with_graph(clips: list[dict], query: str, course_id: str) -> list[dict]:
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
