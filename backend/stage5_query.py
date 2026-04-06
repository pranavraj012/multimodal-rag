import json
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

ROUTING = {
    "concept":    {"targets": ["transcript", "visual"], "granularity": "coarse_then_fine"},
    "visual":     {"targets": ["visual"],               "granularity": "fine_only"},
    "temporal":   {"targets": ["transcript", "visual"], "granularity": "fine_only"},
    "locator":    {"targets": ["transcript", "visual"], "granularity": "coarse_then_fine"},
    "comparison": {"targets": ["transcript", "visual"], "granularity": "coarse_then_fine"},
    "generative": {"targets": ["transcript", "visual"],  "granularity": "coarse_then_fine"},
}

def analyze_query(query: str, history: list) -> dict:
    """
    Uses LLM to rewrite query and classify intent in one shot.
    """
    history_str = "\\n".join([
        f"Turn {i}: Q='{h['query']}' -> clip[{h.get('t_start', '?')}s-{h.get('t_end', '?')}s]"
        for i, h in enumerate(history[-5:])
    ])
    
    prompt = f"""You are a query routing engine for an educational video RAG system.
Analyze the user's current query given the conversation history.

HISTORY:
{history_str}

CURRENT QUERY: "{query}"

Determine the rewritten query (resolving pronouns like "that", "it") and the intent.
Valid intents: "concept", "visual", "temporal", "locator", "comparison", "generative".

For intent:
- concept: explains a topic
- visual: asks to see a diagram/chart/code
- temporal: what came before/after
- locator: where in the video is X
- comparison: difference between X and Y
- generative: summarize, quiz me

Output ONLY a JSON object with keys:
- "rewritten_query": string
- "intent": string (from the valid list)
- "temporal_anchor": null OR {{"t_start": float, "t_end": float}} if it refers to a historic clip
- "content_type_filter": "diagram", "code", or null
- "generation_task": "quiz", "summary", or null

JSON ONLY:"""

    response = call_llm(prompt)
    try:
        clean = response.strip().strip("```json").strip("```").strip()
        result = json.loads(clean)
        
        # Validations
        if result.get("intent") not in ROUTING:
            result["intent"] = "concept"
            
        return result
    except Exception as e:
        print(f"[Stage 5] Fallback routing due to JSON parse error: {e}")
        return {
            "rewritten_query": query, 
            "intent": "concept", 
            "temporal_anchor": None,
            "content_type_filter": None,
            "generation_task": None
        }

def build_routing_object(query: str, history: list) -> RoutingObject:
    res = analyze_query(query, history)
    
    intent = res["intent"]
    config = ROUTING[intent]

    anchor = res.get("temporal_anchor")
    if anchor is None:
        anchor = _fallback_temporal_anchor(query, history)

    generation_task = res.get("generation_task")
    if generation_task == "summary":
        generation_task = _summary_depth_from_query(query)

    return RoutingObject(
        rewritten_query=res["rewritten_query"],
        intent=intent,
        search_targets=config["targets"],
        granularity=config["granularity"],
        temporal_anchor=anchor,
        content_type_filter=res.get("content_type_filter"),
        generation_task=generation_task
    )


def _fallback_temporal_anchor(query: str, history: list) -> Optional[dict]:
    if not history:
        return None

    q = query.lower()
    temporal_markers = [
        "that", "it", "this", "before", "after", "earlier", "later", "previous", "next", "that part"
    ]
    if not any(m in q for m in temporal_markers):
        return None

    last = history[-1]
    if "t_start" in last and "t_end" in last:
        return {
            "t_start": float(last["t_start"]),
            "t_end": float(last["t_end"]),
        }
    return None


def _summary_depth_from_query(query: str) -> str:
    q = query.lower()
    if any(t in q for t in ["one line", "short", "brief", "quick"]):
        return "summary_short"
    if any(t in q for t in ["detailed", "deep", "in detail", "comprehensive"]):
        return "summary_deep"
    return "summary"
