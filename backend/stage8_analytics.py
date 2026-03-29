from stage4_index import get_or_create_collections

def record_clip_rewatch(chunk_id: str, collection_name: str = "transcript_chunks"):
    col, vis_col = get_or_create_collections()
    target = col if collection_name == "transcript_chunks" else vis_col
    result = target.get(ids=[chunk_id], include=["metadatas"])
    if result and result["ids"]:
        meta = result["metadatas"][0]
        meta["rewatch_count"] = meta.get("rewatch_count", 0) + 1
        target.update(ids=[chunk_id], metadatas=[meta])

def get_confusion_report(course_id: str, top_n: int = 10) -> list[dict]:
    col, _ = get_or_create_collections()
    results = col.get(
        where={"course_id": {"$eq": course_id}},
        include=["metadatas", "documents"]
    )
    scored = []
    if results and results["metadatas"]:
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


def get_recommendation_signals(course_id: str, top_n: int = 10) -> list[dict]:
    col, _ = get_or_create_collections()
    results = col.get(
        where={"course_id": {"$eq": course_id}},
        include=["metadatas", "documents"],
    )
    ranked = []
    if results and results.get("metadatas"):
        for meta, doc in zip(results["metadatas"], results["documents"]):
            demand = (
                float(meta.get("query_hit_count", 0)) * 0.6
                + float(meta.get("rewatch_count", 0)) * 0.4
            )
            ranked.append({
                "t_start": meta.get("t_start"),
                "t_end": meta.get("t_end"),
                "lecture_id": meta.get("lecture_id"),
                "recommendation_score": round(demand, 3),
                "text_preview": (doc or "")[:150],
            })

    ranked.sort(key=lambda x: x["recommendation_score"], reverse=True)
    return ranked[:top_n]
