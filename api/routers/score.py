import logging
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from api.models.response import ScoreResponse

log    = logging.getLogger("api.score")
router = APIRouter(prefix="/score", tags=["scoring"])


def _score_to_label(score):
    if score >= 80: return "Very Safe"
    if score >= 60: return "Safe"
    if score >= 40: return "Moderate"
    if score >= 20: return "Unsafe"
    return "Avoid"

def _score_to_grade(score):
    if score >= 80: return "A"
    if score >= 60: return "B"
    if score >= 40: return "C"
    if score >= 20: return "D"
    return "E"

def _score_to_color(score):
    if score >= 80: return "#22c55e"
    if score >= 60: return "#84cc16"
    if score >= 40: return "#f59e0b"
    if score >= 20: return "#ef4444"
    return "#7f1d1d"


@router.get("/", response_model=ScoreResponse)
async def get_score(
    lat:  float = Query(..., ge=-90,  le=90),
    lon:  float = Query(..., ge=-180, le=180),
    hour: int   = Query(22,  ge=0,    le=23),
):
    try:
        from routing.dijkstra import load_graph, find_nearest_node

        G    = load_graph()
        node = find_nearest_node(G, lat, lon)

        edges = list(G.edges(node, data=True))
        if not edges:
            raise HTTPException(status_code=404, detail="No road found nearby.")

        scores = [float(data.get("safety_score", 40.0)) for _, _, data in edges]
        score  = float(np.mean(scores))

        return ScoreResponse(
            lat=lat, lon=lon,
            safety_score=round(score, 1),
            safety_grade=_score_to_grade(score),
            safety_label=_score_to_label(score),
            safety_color=_score_to_color(score),
            hour=hour,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Score error: {e}")
        raise HTTPException(status_code=500, detail=str(e))