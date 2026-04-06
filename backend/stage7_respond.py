from config import call_llm
from stage5_query import RoutingObject
import re

def generate_response(routing: RoutingObject, clips: list[dict], query: str) -> dict:
    if not clips:
        return {
            "answer": "I couldn't find relevant content for that query in this lecture.",
            "clips": []
        }

    transcript_parts = []
    visual_parts = []
    allowed_ranges = []
    for clip in clips[:3]:
        t_label = f"[{clip['t_start']:.0f}s → {clip['t_end']:.0f}s]"
        allowed_ranges.append(t_label)
        text = clip.get("context_text", "").strip()
        vis_desc = clip.get("visual_description", "").strip()
        ocr = clip.get("ocr_text", "").strip()
        if text:
            transcript_parts.append(f"{t_label}: {text[:500]}")
        if vis_desc:
            visual_parts.append(f"{t_label} [on screen]: {vis_desc[:300]}")
        elif ocr:
            visual_parts.append(f"{t_label} [on screen / OCR]: {ocr[:300]}")

    context = "\n\n".join(transcript_parts)
    visual_context = "\n\n".join(visual_parts)

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

    def _build_summary_context() -> str:
        parts = []
        if context.strip():
            parts.append(f"TRANSCRIPT:\n{context}")
        if visual_context.strip():
            parts.append(f"VISUAL FRAMES (what appeared on screen):\n{visual_context}")
        return "\n\n".join(parts) if parts else context

    if routing.generation_task == "quiz":
        answer = call_llm(
            f"Generate 3 multiple choice quiz questions based on both the spoken content and visual frames shown.\n"
            f"Number them 1-3.\n\n{_build_summary_context()}"
        )
    elif routing.generation_task == "summary_short":
        answer = call_llm(
            f"Using both the transcript and any on-screen visuals, summarize this video in exactly 1 concise bullet point.\n\n"
            f"{_build_summary_context()}"
        )
    elif routing.generation_task == "summary_deep":
        answer = call_llm(
            f"Using both what was spoken and what appeared on screen, summarize this video in 6 bullet points:\n"
            "- 3 core ideas\n- 2 important details\n- 1 practical takeaway\n\n"
            f"{_build_summary_context()}"
        )
    elif routing.generation_task == "summary":
        answer = call_llm(
            f"Using both the transcript and any on-screen visuals, summarize this video in exactly 3 bullet points.\n\n"
            f"{_build_summary_context()}"
        )
    else:
        answer = call_llm(
            f"""Answer the student's question using ONLY the lecture content provided below.
If the provided content does NOT contain relevant information to answer the query, reply EXACTLY with "NO_RELEVANT_CONTENT_FOUND".
Be concise (2-4 sentences). If you reference something visual, mention the timestamp.
Only cite timestamps from these ranges: {', '.join(allowed_ranges) if allowed_ranges else 'none'}.
Do not invent new timestamps.

Lecture content:
{context}

Student question: {query}
Answer:"""
        )

    if "NO_RELEVANT_CONTENT_FOUND" in answer:
        return {
            "answer": "I couldn't find relevant content for that query in this lecture.",
            "clips": []
        }

    negative_phrases = [
        "no mention of", "not mentioned", "does not mention", 
        "does not discuss", "not discussed", "no relevant information", 
        "could not find"
    ]
    if any(phrase in answer.lower() for phrase in negative_phrases):
        clips = []

    clips = _prioritize_cited_clip(answer, clips)

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


def _prioritize_cited_clip(answer: str, clips: list[dict]) -> list[dict]:
    if not answer or not clips:
        return clips

    sec_hints = [int(m.group(1)) for m in re.finditer(r"\b(\d{1,5})s\b", answer)]
    for m in re.finditer(r"\b(\d{1,2}):(\d{2})\b", answer):
        sec_hints.append(int(m.group(1)) * 60 + int(m.group(2)))

    # Preserve order while removing duplicates.
    sec_hints = list(dict.fromkeys(sec_hints))
    if not sec_hints:
        return clips

    target = sec_hints[0]

    def distance_to_clip(c: dict) -> float:
        start = float(c.get("t_start", 0.0))
        end = float(c.get("t_end", start))
        if start <= target <= end:
            return 0.0
        return min(abs(target - start), abs(target - end))

    best_clip = min(clips, key=distance_to_clip)
    if distance_to_clip(best_clip) > 90:
        return clips

    # Keep original relative order for remaining clips.
    rest = [c for c in clips if c is not best_clip]
    return [best_clip] + rest
