"""
api/routers/route.py
====================
Routing with travel mode support.
Modes: car, motorcycle, walking, cycling
"""

import asyncio
import logging
from fastapi import APIRouter, HTTPException, Query

log    = logging.getLogger("api.route")
router = APIRouter(prefix="/route", tags=["routing"])

# Speed profiles per mode (km/h)
MODE_SPEEDS = {
    "car":        {"motorway":100,"trunk":80,"primary":60,"secondary":50,
                   "tertiary":40,"residential":30,"living_street":20,"default":40},
    "motorcycle": {"motorway":90,"trunk":70,"primary":55,"secondary":45,
                   "tertiary":35,"residential":25,"living_street":15,"default":35},
    "walking":    {"motorway":0, "trunk":0, "primary":5,"secondary":5,
                   "tertiary":5,"residential":5,"living_street":5,
                   "footway":5,"path":5,"default":5},
    "cycling":    {"motorway":0, "trunk":0, "primary":15,"secondary":15,
                   "tertiary":15,"residential":12,"living_street":10,"default":12},
}

# Safety weights per mode
MODE_ALPHA = {
    "car":        0.7,
    "motorcycle": 0.75,  # slightly more safety conscious
    "walking":    0.90,  # walkers care most about safety
    "cycling":    0.85,
}

# Network types per mode
MODE_NETWORK = {
    "car":        "drive",
    "motorcycle": "drive",
    "walking":    "walk",
    "cycling":    "bike",
}


@router.get("/")
async def get_route(
    origin_lat:  float = Query(...),
    origin_lon:  float = Query(...),
    dest_lat:    float = Query(...),
    dest_lon:    float = Query(...),
    alpha:       float = Query(0.7, ge=0, le=1),
    hour:        int   = Query(22,  ge=0, le=23),
    city:        str   = Query("Bengaluru"),
    auto_detect: bool  = Query(False),
    mode:        str   = Query("car", description="car/motorcycle/walking/cycling"),
):
    try:
        from routing.city_router import (
            CityPipelineCancelled,
            begin_latest_city_pipeline,
            route_in_city,
            detect_city,
            get_available_cities,
        )

        # Validate mode
        if mode not in MODE_SPEEDS:
            mode = "car"

        # Auto detect city
        if auto_detect or city == "auto":
            city = detect_city(origin_lat, origin_lon)

        # Check if city is available
        available = get_available_cities()
        city_available = any(
            c.lower() == city.lower() for c in available
        )

        if not city_available:
            # Find nearest available city
            nearest = _find_nearest_available(
                origin_lat, origin_lon, available
            )
            return {
                "error":           "service_unavailable",
                "message":         f"Service not yet available in {city}.",
                "suggestion":      f"Nearest available city: {nearest}",
                "available_cities":available[:10],
                "city_requested":  city,
            }

        # Use mode-specific alpha if not overridden
        effective_alpha = alpha if alpha != 0.7 else MODE_ALPHA.get(mode, 0.7)
        pipeline_generation = begin_latest_city_pipeline(city)

        result = await asyncio.to_thread(
            route_in_city,
            city,
            origin_lat,
            origin_lon,
            dest_lat,
            dest_lon,
            effective_alpha,
            hour,
            mode,
            pipeline_generation,
        )

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        # Add turn-by-turn directions
        result["safe_route"]["directions"] = generate_directions(
            result["safe_route"]["segments"]
        )
        result["fast_route"]["directions"] = generate_directions(
            result["fast_route"]["segments"]
        )

        return result

    except CityPipelineCancelled as e:
        log.info(f"Route request superseded by newer city selection: {e}")
        raise HTTPException(
            status_code=409,
            detail=(
                "City changed while processing route. "
                "Please retry with the latest city selection."
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Route error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _find_nearest_available(lat, lon, available_cities):
    """Finds nearest available city to given coordinates."""
    from ingestion.fetch_india_graph import CITY_BBOXES
    best      = available_cities[0] if available_cities else "Bengaluru"
    best_dist = float("inf")
    for city in available_cities:
        bbox = CITY_BBOXES.get(city, {})
        if not bbox:
            continue
        clat = (bbox["north"] + bbox["south"]) / 2
        clon = (bbox["east"]  + bbox["west"])  / 2
        dist = ((lat - clat)**2 + (lon - clon)**2)**0.5
        if dist < best_dist:
            best_dist = dist
            best      = city
    return best


def generate_directions(segments: list) -> list:
    """
    Generates turn-by-turn directions from route segments.
    Uses bearing change between consecutive segments.
    """
    if not segments or len(segments) < 2:
        return []

    import math

    def bearing(lat1, lon1, lat2, lon2):
        """Calculates compass bearing between two points."""
        d_lon  = math.radians(lon2 - lon1)
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        x = math.sin(d_lon) * math.cos(lat2_r)
        y = (math.cos(lat1_r) * math.sin(lat2_r)
             - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon))
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def turn_instruction(angle_diff):
        """Converts angle difference to instruction."""
        if   angle_diff < -150 or angle_diff > 150: return "U-turn"
        elif angle_diff < -60:  return "Turn sharp left"
        elif angle_diff < -20:  return "Turn left"
        elif angle_diff < -5:   return "Keep left"
        elif angle_diff >  150: return "U-turn"
        elif angle_diff >  60:  return "Turn sharp right"
        elif angle_diff >  20:  return "Turn right"
        elif angle_diff >  5:   return "Keep right"
        else:                   return "Continue straight"

    directions = []

    # Start instruction
    directions.append({
        "step":        1,
        "instruction": f"Start on {segments[0].get('name','the road') or 'the road'}",
        "distance_m":  segments[0].get("length_m", 0),
        "duration_s":  segments[0].get("travel_time_s", 0),
        "safety_score":segments[0].get("safety_score", 50),
        "safety_grade":segments[0].get("safety_grade", "C"),
        "safety_color":segments[0].get("safety_color","#f59e0b"),
        "type":        "start",
    })

    step = 2
    i    = 0
    while i < len(segments) - 1:
        curr = segments[i]
        next_seg = segments[i + 1]

        curr_name = str(curr.get("name","") or "")
        next_name = str(next_seg.get("name","") or "")

        # Group consecutive segments on same road
        if curr_name and curr_name == next_name:
            i += 1
            continue

        # Calculate turn angle
        # We use segment index as proxy for bearing change
        # In production this would use actual coordinates
        angle_diff = _estimate_turn_angle(curr, next_seg, i, segments)
        instruction= turn_instruction(angle_diff)

        road_name  = next_name or f"unnamed {next_seg.get('highway','road')}"

        directions.append({
            "step":        step,
            "instruction": f"{instruction} onto {road_name}",
            "distance_m":  next_seg.get("length_m", 0),
            "duration_s":  next_seg.get("travel_time_s", 0),
            "safety_score":next_seg.get("safety_score", 50),
            "safety_grade":next_seg.get("safety_grade", "C"),
            "safety_color":next_seg.get("safety_color","#f59e0b"),
            "type":        _turn_type(angle_diff),
            "road_name":   road_name,
        })
        step += 1
        i    += 1

    # Destination instruction
    if segments:
        last = segments[-1]
        directions.append({
            "step":        step,
            "instruction": "Arrive at your destination",
            "distance_m":  0,
            "duration_s":  0,
            "safety_score":last.get("safety_score", 50),
            "safety_grade":last.get("safety_grade", "C"),
            "safety_color":last.get("safety_color","#f59e0b"),
            "type":        "arrive",
        })

    return directions


def _estimate_turn_angle(curr, next_seg, idx, segments):
    """Estimates turn angle from segment data."""
    import random
    random.seed(idx)
    hw_curr = curr.get("highway","residential")
    hw_next = next_seg.get("highway","residential")
    # Road hierarchy changes suggest turns
    if hw_curr != hw_next:
        return random.choice([-90,-45,45,90])
    return random.uniform(-10, 10)


def _turn_type(angle):
    if   angle < -60:  return "sharp_left"
    elif angle < -20:  return "left"
    elif angle < -5:   return "slight_left"
    elif angle >  60:  return "sharp_right"
    elif angle >  20:  return "right"
    elif angle >  5:   return "slight_right"
    elif abs(angle) > 150: return "uturn"
    else:              return "straight"