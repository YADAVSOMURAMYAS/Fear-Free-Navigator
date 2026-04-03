import logging
import random
import numpy as np
from fastapi import APIRouter, Query
from api.models.response import HeatmapResponse

log    = logging.getLogger("api.heatmap")
router = APIRouter(prefix="/heatmap", tags=["heatmap"])


@router.get("/", response_model=HeatmapResponse)
async def get_heatmap(
    sample_n: int = Query(3000, ge=100, le=10000),
):
    try:
        from routing.dijkstra import load_graph
        G      = load_graph()
        points = []

        for u, v, data in G.edges(data=True):
            u_lat = G.nodes[u]["y"]
            u_lon = G.nodes[u]["x"]
            v_lat = G.nodes[v]["y"]
            v_lon = G.nodes[v]["x"]
            score = float(data.get("safety_score", 40.0))
            points.append([
                round((u_lat + v_lat) / 2, 6),
                round((u_lon + v_lon) / 2, 6),
                round(score, 1),
            ])

        if len(points) > sample_n:
            random.seed(42)
            points = random.sample(points, sample_n)

        return HeatmapResponse(points=points, count=len(points))

    except Exception as e:
        log.error(f"Heatmap error: {e}")
        return HeatmapResponse(points=[], count=0)