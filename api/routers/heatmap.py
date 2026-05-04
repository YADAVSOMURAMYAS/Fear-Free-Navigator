"""
api/routers/heatmap.py
======================
Safety heatmap endpoint — works for all 50 cities.
"""

import asyncio
import logging
import random
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from api.models.response import HeatmapResponse

log    = logging.getLogger("api.heatmap")
router = APIRouter(prefix="/heatmap", tags=["heatmap"])


@router.get("/", response_model=HeatmapResponse)
async def get_heatmap(
    sample_n:  int = Query(3000, ge=100, le=10000),
    city:      str = Query("Bengaluru"),
):
    try:
        log.info(f"🔥 Heatmap Request: {city} | Samples: {sample_n}")
        
        from routing.city_router import (
            CityPipelineCancelled,
            begin_latest_city_pipeline,
            load_city_graph,
        )
        log.info(f"📊 Loading city graph for {city}...")
        pipeline_generation = begin_latest_city_pipeline(city)
        G = await asyncio.to_thread(
            load_city_graph,
            city,
            pipeline_generation,
        )
        log.info(f"   ✓ Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
        log.info(f"🗺️  Extracting safety scores from edges...")
        points = []

        for u, v, data in G.edges(data=True):
            try:
                u_lat = float(G.nodes[u]["y"])
                u_lon = float(G.nodes[u]["x"])
                v_lat = float(G.nodes[v]["y"])
                v_lon = float(G.nodes[v]["x"])
            except (KeyError, ValueError):
                continue

            score = float(data.get(
                "temporal_safety",
                data.get("safety_score", 40.0)
            ))
            try:
                score = float(score)
            except:
                score = 40.0

            points.append([
                round((u_lat + v_lat) / 2, 6),
                round((u_lon + v_lon) / 2, 6),
                round(score, 1),
            ])

        log.info(f"   ✓ Extracted {len(points)} safety points")

        # Sample for performance
        if len(points) > sample_n:
            log.info(f"📉 Sampling {sample_n} points from {len(points)} (for performance)...")
            random.seed(42)
            points = random.sample(points, sample_n)

        log.info(f"✅ Heatmap complete: {len(points)} points ready")
        return HeatmapResponse(points=points, count=len(points))

    except CityPipelineCancelled as e:
        log.info(f"Heatmap request superseded by newer city selection: {e}")
        raise HTTPException(
            status_code=409,
            detail=(
                "City changed while processing heatmap. "
                "Please retry with the latest city selection."
            ),
        )

    except Exception as e:
        log.error(f"Heatmap error: {e}")
        return HeatmapResponse(points=[], count=0)