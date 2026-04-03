"""
api/routers/route.py
====================
Main routing endpoint.
GET /route → returns safe + fast dual routes
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from api.models.request  import RouteRequest
from api.models.response import DualRouteResponse

log    = logging.getLogger("api.route")
router = APIRouter(prefix="/route", tags=["routing"])


@router.get("/")
async def get_route(
    origin_lat: float = Query(..., description="Origin latitude"),
    origin_lon: float = Query(..., description="Origin longitude"),
    dest_lat:   float = Query(..., description="Destination latitude"),
    dest_lon:   float = Query(..., description="Destination longitude"),
    alpha:      float = Query(0.7, ge=0, le=1, description="Safety weight"),
    hour:       int   = Query(22,  ge=0, le=23,description="Hour of day"),
):
    """
    Returns dual routes: safest path + fastest path.

    - **alpha=1.0** → pure safety routing
    - **alpha=0.0** → pure speed routing
    - **alpha=0.7** → recommended (safety-weighted)
    """
    try:
        from routing.dijkstra import get_dual_routes
        result = get_dual_routes(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            alpha=alpha,
            hour=hour,
        )

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def post_route(req: RouteRequest):
    """Same as GET /route but accepts JSON body."""
    try:
        from routing.dijkstra import get_dual_routes
        result = get_dual_routes(
            origin_lat=req.origin_lat,
            origin_lon=req.origin_lon,
            dest_lat=req.dest_lat,
            dest_lon=req.dest_lon,
            alpha=req.alpha,
            hour=req.hour,
        )

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))