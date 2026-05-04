"""
api/routers/cities.py
=====================
City management endpoints.
"""

import logging
from fastapi import APIRouter, Query

log    = logging.getLogger("api.cities")
router = APIRouter(prefix="/cities", tags=["cities"])


@router.get("/")
async def get_cities():
    """Returns list of all cities with downloaded graphs."""
    try:
        log.info(f"📍 Fetching available cities...")
        from routing.city_router import get_available_cities
        cities = get_available_cities()
        log.info(f"   ✓ Found {len(cities)} cities: {', '.join(cities[:5])}{'...' if len(cities) > 5 else ''}")
        return {
            "cities":  cities,
            "total":   len(cities),
            "default": "Bengaluru",
        }
    except Exception as e:
        log.error(f"❌ Cities error: {e}")
        return {
            "cities":  ["Bengaluru"],
            "total":   1,
            "default": "Bengaluru",
        }


@router.get("/detect")
async def detect_city(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
):
    """Auto-detects city from GPS coordinates."""
    try:
        from routing.city_router import detect_city as _detect
        city = _detect(lat, lon)
        return {"city": city, "lat": lat, "lon": lon}
    except Exception as e:
        log.error(f"Detect error: {e}")
        return {"city": "Bengaluru", "lat": lat, "lon": lon}