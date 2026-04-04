"""
ai/llm/explainer.py
===================
Uses Claude API to generate plain-English safety explanations
for each road segment and full route summary.

This is the UNIQUE differentiator — no other navigation app
explains WHY a route is safe or unsafe in human language.

Run test:
    python -m ai.llm.explainer
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from functools import lru_cache

from prompt_toolkit import prompt
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("llm.explainer")

CACHE_DIR = Path("ai/llm/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a safety analyst for a night navigation app in Bengaluru, India.
Your job is to explain why a road segment or route got its safety score.
Speak directly to a solo traveller or woman commuting at night.
Be specific, honest, empathetic, and concise.
Always respond with valid JSON only. No markdown, no preamble."""

SEGMENT_PROMPT_TEMPLATE = """
Analyze this Bengaluru road segment for night safety:

Road: {road_name} ({highway_type})
Safety Score: {safety_score}/100 (Grade {safety_grade})
Hour: {hour}:00

Key factors:
- Lighting: {lighting_desc}
- Commercial activity: {commercial_desc}  
- Police/Emergency: {emergency_desc}
- Crime history: {crime_desc}
- Road type: {road_type_desc}
- Visual environment: {visual_desc}

Respond with this exact JSON:
{{
  "explanation": "2-3 sentence explanation of why this score",
  "top_risk": "single biggest safety concern",
  "top_positive": "single biggest safety positive",
  "advice": "one practical tip for travelling this segment at night"
}}"""

ROUTE_PROMPT_TEMPLATE = """
Summarize this safe route recommendation for a night traveller in Bengaluru:

Route: {origin_name} → {destination_name}
Time: {hour}:00 ({time_period})
Safe Route Score: {safe_score}/100 (Grade {safe_grade})
Fast Route Score: {fast_score}/100 (Grade {fast_grade})
Safety Gain: +{safety_gain} points
Time Cost: +{time_cost} minutes
Key roads: {road_names}
Dangerous segments avoided: {dangerous_avoided}

Respond with this exact JSON:
{{
  "summary": "2-3 sentence summary of why the safe route is recommended",
  "key_benefit": "the single most important safety benefit",
  "time_verdict": "is the time cost worth it?",
  "confidence": "high/medium/low"
}}"""


# ──────────────────────────────────────────────────────────────────────────────
# 2.  CACHE SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


def _load_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(key: str, data: dict):
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(data, f)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CLAUDE API CALL
# ──────────────────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, max_tokens: int = 300) -> dict | None:
    """
    Calls Groq API (llama3-8b-8192) for fast, free LLM inference.
    Groq is 10x faster than Claude for this use case.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        log.warning("GROQ_API_KEY not set. Using fallback explanation.")
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama3-8b-8192",   # fast + free
            max_tokens=max_tokens,
            temperature=0.3,           # low temp = consistent JSON
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown if model wraps in ```json
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e} | Raw: {raw[:100]}")
        return None
    except Exception as e:
        log.error(f"Groq API error: {e}")
        return None
# ──────────────────────────────────────────────────────────────────────────────
# 4.  SEGMENT EXPLANATION
# ──────────────────────────────────────────────────────────────────────────────

def _build_lighting_desc(luminosity_score: float, lamp_count: int) -> str:
    if luminosity_score > 70:
        return f"Well-lit area (score {luminosity_score:.0f}/100, {lamp_count} lamps nearby)"
    elif luminosity_score > 40:
        return f"Moderate lighting (score {luminosity_score:.0f}/100)"
    else:
        return f"Poor lighting (score {luminosity_score:.0f}/100) — dark at night"


def _build_commercial_desc(commercial_score: float, hour: int) -> str:
    is_night = hour >= 20 or hour <= 5
    if commercial_score > 0.7:
        activity = "active" if not is_night else "likely closed at this hour"
        return f"High commercial density — {activity}"
    elif commercial_score > 0.4:
        return f"Moderate shops and restaurants nearby"
    else:
        return f"Few commercial establishments — isolated area"


def _build_crime_desc(crime_density: float, night_crime: float) -> str:
    if night_crime > 0.6:
        return f"High night crime density in this zone (risk: {night_crime:.0%})"
    elif crime_density > 0.4:
        return f"Moderate historical crime incidents nearby"
    else:
        return f"Low historical crime rate in this area"


def explain_segment(
    segment_data: dict,
    hour: int = 22,
    use_cache: bool = True,
) -> dict:
    """
    Generates plain-English explanation for a road segment's safety score.

    Args:
        segment_data: dict with safety features (from feature store)
        hour: hour of day

    Returns:
        dict with explanation, top_risk, top_positive, advice
    """
    score       = float(segment_data.get("safety_score", 50))
    grade       = segment_data.get("safety_grade", "C")
    road_name   = segment_data.get("name", "Unknown Road") or "Unnamed Road"
    highway     = segment_data.get("highway", "road")
    lum_score   = float(segment_data.get("luminosity_score", 35))
    lamp_count  = int(segment_data.get("lamp_count_80m", 0))
    comm_score  = float(segment_data.get("commercial_score", 0.3))
    police_n    = int(segment_data.get("police_count_500m", 0))
    crime_d     = float(segment_data.get("crime_density", 0.2))
    night_crime = float(segment_data.get("night_crime_density", 0.25))
    visual      = float(segment_data.get("visual_score", 0.5))
    is_primary  = bool(segment_data.get("is_primary_secondary", 0))

    # Cache check
    cache_key = _cache_key(
        f"{road_name}{score:.0f}{hour}{lum_score:.0f}{crime_d:.2f}"
    )
    if use_cache:
        cached = _load_cache(cache_key)
        if cached:
            return cached

    # Build prompt
    prompt = SEGMENT_PROMPT_TEMPLATE.format(
        road_name     = road_name,
        highway_type  = highway,
        safety_score  = f"{score:.1f}",
        safety_grade  = grade,
        hour          = hour,
        lighting_desc = _build_lighting_desc(lum_score, lamp_count),
        commercial_desc=_build_commercial_desc(comm_score, hour),
        emergency_desc= f"{police_n} police station(s) within 500m" if police_n else "No police stations nearby",
        crime_desc    = _build_crime_desc(crime_d, night_crime),
        road_type_desc= f"{'Major road (high footfall)' if is_primary else 'Minor road (low footfall)'}",
        visual_desc   = f"Visual safety score {visual:.2f}/1.0",
    )

    result = _call_llm(prompt)  
    if result is None:
        result = _fallback_segment_explanation(score, road_name, hour, crime_d, lum_score)

    if use_cache:
        _save_cache(cache_key, result)

    return result


def _fallback_segment_explanation(
    score: float,
    road_name: str,
    hour: int,
    crime_density: float,
    luminosity: float,
) -> dict:
    """Rule-based fallback when Claude API unavailable."""
    if score >= 70:
        return {
            "explanation": f"{road_name} is a well-lit, commercially active road with good safety infrastructure. Historical crime data shows low risk in this area.",
            "top_risk": "Monitor for reduced activity late at night when shops close.",
            "top_positive": "Good street lighting and commercial presence.",
            "advice": "This is a relatively safe route. Stay on the main road.",
        }
    elif score >= 50:
        return {
            "explanation": f"{road_name} has moderate safety conditions. Lighting is present but crime history suggests some caution is warranted.",
            "top_risk": "Moderate crime density in this zone." if crime_density > 0.3 else "Limited lighting in sections.",
            "top_positive": "Road has some commercial activity and transit connections.",
            "advice": "Stay alert and prefer the busier lanes of this road.",
        }
    else:
        return {
            "explanation": f"{road_name} scores low on safety. Poor lighting combined with limited commercial activity makes this road concerning for solo night travel.",
            "top_risk": "Poor lighting and isolated stretches." if luminosity < 30 else "High crime density in surrounding area.",
            "top_positive": "The route is relatively short.",
            "advice": "Avoid this road after dark if possible. If necessary, share your location with someone.",
        }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  ROUTE EXPLANATION
# ──────────────────────────────────────────────────────────────────────────────

def explain_route(
    route_result: dict,
    origin_name: str = "Origin",
    dest_name:   str = "Destination",
    use_cache:   bool = True,
) -> dict:
    """
    Generates plain-English explanation for why the safe route
    is recommended over the fast route.
    """
    safe = route_result.get("safe_route", {})
    fast = route_result.get("fast_route", {})
    comp = route_result.get("comparison", {})
    hour = route_result.get("hour", 22)

    safe_score  = safe.get("avg_safety_score", 50)
    fast_score  = fast.get("avg_safety_score", 50)
    safety_gain = comp.get("safety_gain_points", 0)
    time_cost   = comp.get("time_penalty_min", 0)

    # Get unique road names from safe route
    segments   = safe.get("segments", [])
    road_names = list(dict.fromkeys([
        s["name"] for s in segments
        if s.get("name") and s["name"] != "Unknown Road"
    ]))[:5]

    # Count dangerous segments avoided
    fast_danger = fast.get("dangerous_count", 0)
    safe_danger = safe.get("dangerous_count", 0)
    avoided     = max(0, fast_danger - safe_danger)

    time_period = (
        "late night" if hour >= 22 or hour <= 4 else
        "night"      if hour >= 20 else
        "evening"    if hour >= 17 else
        "daytime"
    )

    cache_key = _cache_key(
        f"{origin_name}{dest_name}{safe_score:.0f}{fast_score:.0f}{hour}"
    )
    if use_cache:
        cached = _load_cache(cache_key)
        if cached:
            return cached

    prompt = ROUTE_PROMPT_TEMPLATE.format(
        origin_name      = origin_name,
        destination_name = dest_name,
        hour             = hour,
        time_period      = time_period,
        safe_score       = f"{safe_score:.1f}",
        safe_grade       = safe.get("safety_grade", "B"),
        fast_score       = f"{fast_score:.1f}",
        fast_grade       = fast.get("safety_grade", "C"),
        safety_gain      = f"{safety_gain:.1f}",
        time_cost        = f"{time_cost:.1f}",
        road_names       = ", ".join(road_names) if road_names else "city roads",
        dangerous_avoided= avoided,
    )

    result = _call_llm(prompt, max_tokens=400)

    if result is None:
        result = {
            "summary": f"The safe route scores {safe_score:.1f}/100 vs {fast_score:.1f}/100 for the fast route, gaining {safety_gain:.1f} safety points for just {time_cost:.1f} extra minutes.",
            "key_benefit": "Better lighting and lower crime density throughout.",
            "time_verdict": "Worth it." if time_cost <= 5 else "Significant time cost — your choice.",
            "confidence": "high" if safety_gain >= 10 else "medium",
        }

    if use_cache:
        _save_cache(cache_key, result)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 6.  BATCH EXPLAIN TOP DANGEROUS SEGMENTS
# ──────────────────────────────────────────────────────────────────────────────

def explain_dangerous_segments(
    route_result: dict,
    hour: int = 22,
    top_n: int = 3,
) -> list:
    """
    Returns explanations for the top N most dangerous segments
    in the fast route that the safe route avoids.
    Used to show users exactly what dangers they avoided.
    """
    fast_segments = route_result.get("fast_route", {}).get("segments", [])
    safe_node_pairs = {
        (s["u"], s["v"])
        for s in route_result.get("safe_route", {}).get("segments", [])
    }

    # Find segments in fast route but NOT in safe route
    avoided = [
        s for s in fast_segments
        if (s["u"], s["v"]) not in safe_node_pairs
        and s["safety_score"] < 50
    ]

    # Sort by safety score ascending (worst first)
    avoided.sort(key=lambda x: x["safety_score"])
    avoided = avoided[:top_n]

    explanations = []
    for seg in avoided:
        exp = explain_segment(seg, hour=hour)
        explanations.append({
            "road_name":    seg.get("name", "Unknown"),
            "safety_score": seg["safety_score"],
            "safety_grade": seg["safety_grade"],
            **exp,
        })

    return explanations


# ──────────────────────────────────────────────────────────────────────────────
# 7.  TEST
# ──────────────────────────────────────────────────────────────────────────────

def run_test():
    logging.basicConfig(level=logging.INFO)
    log.info("=== LLM Explainer Test ===")

    # Test segment explanation
    test_segment = {
        "safety_score":      45.0,
        "safety_grade":      "C",
        "name":              "Chickpete Road",
        "highway":           "secondary",
        "luminosity_score":  28.0,
        "lamp_count_80m":    1,
        "commercial_score":  0.3,
        "police_count_500m": 0,
        "crime_density":     0.65,
        "night_crime_density": 0.72,
        "visual_score":      0.3,
        "is_primary_secondary": 0,
    }

    print("\n── Segment Explanation ──")
    result = explain_segment(test_segment, hour=23)
    print(json.dumps(result, indent=2))

    log.info("=== Test DONE ===")
    return result


if __name__ == "__main__":
    run_test()