from config import call_llm
from stage5_query import RoutingObject

def generate_response(routing: RoutingObject, clips: list[dict], query: str) -> dict:
    if not clips:
        return {
            "answer": "I couldn't find relevant content for that query in this lecture.",
            "clips": []
        }

    context_parts = []
    for clip in clips[:3]:
        text = clip.get("context_text", "").strip()
        if text:
            context_parts.append(f"[{clip['t_start']:.0f}s → {clip['t_end']:.0f}s]: {text[:500]}")
    context = "\\n\\n".join(context_parts)

    if routing.intent == "visual" and len(context.strip()) < 40:
        moments = ", ".join([f"[{c['t_start']:.0f}s -> {c['t_end']:.0f}s]" for c in clips[:3]])
        answer = (
            "I found likely visual hotspots in the lecture, but extracted visual text is limited. "
            f"Start with these moments: {moments}."
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

    if routing.generation_task == "quiz":
        answer = call_llm(f"Generate 3 multiple choice quiz questions from this lecture content. Number them 1-3.\\n\\n{context}")
    elif routing.generation_task == "summary_short":
        answer = call_llm(f"Summarize this lecture content in exactly 1 concise bullet point.\\n\\n{context}")
    elif routing.generation_task == "summary_deep":
        answer = call_llm(
            f"Summarize this lecture content in 6 bullet points with:\n"
            "- 3 core ideas\n"
            "- 2 important details\n"
            "- 1 practical takeaway\n\n"
            f"{context}"
        )
    elif routing.generation_task == "summary":
        answer = call_llm(f"Summarize this lecture content in exactly 3 bullet points.\\n\\n{context}")
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
