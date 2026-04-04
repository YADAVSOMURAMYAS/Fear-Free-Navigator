"""
routing/dijkstra.py
===================
Modified Dijkstra routing engine that optimizes for
safety score instead of just travel time.

Core algorithm:
    edge_weight = alpha * (1/safety_score) + (1-alpha) * travel_time

    alpha = 1.0 → pure safety (ignore time)
    alpha = 0.0 → pure speed  (ignore safety)
    alpha = 0.7 → recommended (safety-weighted)

Run test:
    python -m routing.dijkstra

Depends on:
    data/processed/bengaluru_scored_graph.graphml
    ai/ml/artifacts/safety_model.pkl
"""

import logging
import json
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import joblib
from pathlib import Path
from functools import lru_cache
from typing import Optional

log = logging.getLogger("routing.dijkstra")

DATA_PROCESSED = Path("data/processed")
ARTIFACTS      = Path("ai/ml/artifacts")

DEFAULT_ALPHA    = 0.7
MIN_SAFETY_SCORE = 1.0
DEFAULT_HOUR     = 22


# ──────────────────────────────────────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────────────────────────────────────

def _sf(val, default: float) -> float:
    """Safe float conversion — graphml stores all values as strings."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH LOADING
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_graph():
    scored_path = DATA_PROCESSED / "bengaluru_scored_graph.graphml"
    base_path   = Path("data/raw/bengaluru_graph.graphml")

    if scored_path.exists():
        log.info(f"Loading scored graph: {scored_path}")
        G = ox.load_graphml(scored_path)
    elif base_path.exists():
        log.info(f"Loading base graph: {base_path}")
        G = ox.load_graphml(base_path)
    else:
        raise FileNotFoundError("No graph found. Run ingestion pipeline first.")

    log.info(f"Graph loaded: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
    return G


def inject_ml_scores(G, hour: int = 22):
    ml_path = DATA_PROCESSED / "bengaluru_feature_store_ml.csv"
    if not ml_path.exists():
        log.warning("ML scores not found. Using default score 40.")
        for u, v, k, data in G.edges(data=True, keys=True):
            data["safety_score"] = 40.0
        return G

    log.info("Injecting ML safety scores into graph edges ...")
    df = pd.read_csv(ml_path)

    score_lookup = {}
    for _, row in df.iterrows():
        key = (str(row["u"]), str(row["v"]), str(row["key"]))
        score_lookup[key] = _sf(row.get("safety_score_ml", 40.0), 40.0)

    injected = 0
    missing  = 0
    for u, v, k, data in G.edges(data=True, keys=True):
        lookup_key = (str(u), str(v), str(k))
        score = score_lookup.get(lookup_key)
        if score is not None:
            data["safety_score"] = score
            injected += 1
        else:
            data["safety_score"] = 40.0
            missing += 1

    log.info(f"Injected: {injected:,} | Default: {missing:,}")
    return G


# ──────────────────────────────────────────────────────────────────────────────
# 2.  WEIGHT COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_edge_weights(
    G,
    alpha: float = DEFAULT_ALPHA,
    hour:  int   = DEFAULT_HOUR,
):
    """
    Time-aware edge weights with aggressive night penalty.

    At night (20:00-06:00):
      - Crime penalty is EXPONENTIAL (6x at 2AM)
      - Dark roads get hard lighting floor penalty
      - Commercial activity collapse penalty
    """
    if   6  <= hour < 17: period = "day"
    elif 17 <= hour < 20: period = "evening"
    elif 20 <= hour < 24: period = "night"
    else:                 period = "late_night"

    # Exponential crime multipliers
    CRIME_EXP = {
        "day":        1.0,
        "evening":    1.8,
        "night":      3.5,
        "late_night": 6.0,
    }
    # Dark road penalty at night
    LIGHTING_FLOOR = {
        "day":        0.0,
        "evening":    5.0,
        "night":      18.0,
        "late_night": 30.0,
    }
    # Commercial collapse penalty
    COMMERCIAL_PENALTY = {
        "day":        0.0,
        "evening":    3.0,
        "night":      12.0,
        "late_night": 20.0,
    }

    ce = CRIME_EXP[period]
    lf = LIGHTING_FLOOR[period]
    cp = COMMERCIAL_PENALTY[period]

    times   = [_sf(d.get("travel_time", 60), 60) for _, _, d in G.edges(data=True)]
    max_t   = max(times) if times else 300
    min_t   = min(times) if times else 1
    range_t = max(max_t - min_t, 1)

    for u, v, key, data in G.edges(data=True, keys=True):

        base_score = float(np.clip(
            _sf(data.get("safety_score",     40.0), 40.0), 0, 100
        ))
        luminosity  = _sf(data.get("luminosity_score", 35.0), 35.0)
        crime       = _sf(data.get("crime_density",     0.15), 0.15)
        commercial  = _sf(data.get("commercial_score",   0.3),  0.3)
        lum_norm    = float(np.clip(luminosity / 100.0, 0, 1))

        # 1. Exponential crime penalty
        crime_pen = float((crime ** 0.5) * 15 * ce)

        # 2. Darkness penalty — only at night
        dark_pen = float(max(0.0, (1.0 - lum_norm) - 0.35) * lf)

        # 3. Commercial collapse penalty
        comm_pen = float(max(0.0, 0.6 - commercial) * cp)

        # Temporal safety score clamped 1-100
        temporal_safety = float(np.clip(
            base_score - crime_pen - dark_pen - comm_pen,
            1.0, 100.0
        ))

        travel_time = _sf(data.get("travel_time", 60.0), 60.0)
        time_norm   = (travel_time - min_t) / range_t * 100

        data["temporal_safety"]  = round(temporal_safety, 2)
        data["composite_weight"] = (
            alpha * (100.0 / max(temporal_safety, MIN_SAFETY_SCORE)) +
            (1 - alpha) * time_norm
        )

    return G


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ROUTE COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def find_nearest_node(G, lat: float, lon: float) -> int:
    return ox.nearest_nodes(G, lon, lat)


def compute_route(
    G,
    orig_node: int,
    dest_node: int,
    weight: str = "composite_weight",
) -> Optional[list]:
    try:
        return nx.shortest_path(G, orig_node, dest_node, weight=weight)
    except nx.NetworkXNoPath:
        log.warning(f"No path found from {orig_node} to {dest_node}")
        return None
    except Exception as e:
        log.error(f"Routing error: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ROUTE STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_route_stats(G, route: list, hour: int = 22) -> dict:
    if not route or len(route) < 2:
        return {}

    edge_scores  = []
    edge_times   = []
    edge_lengths = []
    segments     = []
    coords       = []

    for node in route:
        coords.append([
            _sf(G.nodes[node]["y"], 12.97),
            _sf(G.nodes[node]["x"], 77.59),
        ])

    for u, v in zip(route, route[1:]):
        data = G[u][v][0]

        score  = _sf(
            data.get("temporal_safety", data.get("safety_score", 40.0)),
            40.0
        )
        ttime  = _sf(data.get("travel_time", 60.0), 60.0)
        length = _sf(data.get("length",      50.0), 50.0)

        hw = data.get("highway", "unknown")
        if isinstance(hw, list):
            hw = hw[0]
        hw = str(hw)

        edge_scores.append(score)
        edge_times.append(ttime)
        edge_lengths.append(length)

        segments.append({
            "u":             u,
            "v":             v,
            "safety_score":  round(score,  1),
            "travel_time_s": round(ttime,  1),
            "length_m":      round(length, 1),
            "highway":       hw,
            "name":          str(data.get("name", "") or ""),
            "safety_grade":  _score_to_grade(score),
            "safety_color":  _score_to_color(score),
        })

    total_time_min = sum(edge_times)   / 60
    total_dist_km  = sum(edge_lengths) / 1000
    avg_safety     = float(np.mean(edge_scores))
    min_safety     = float(np.min(edge_scores))
    dangerous      = [s for s in segments if s["safety_score"] < 30]

    return {
        "coords":            coords,
        "segments":          segments,
        "avg_safety_score":  round(avg_safety,     1),
        "min_safety_score":  round(min_safety,     1),
        "total_time_min":    round(total_time_min, 1),
        "total_dist_km":     round(total_dist_km,  2),
        "n_segments":        len(segments),
        "dangerous_count":   len(dangerous),
        "hour":              hour,
        "safety_grade":      _score_to_grade(avg_safety),
    }


def _score_to_grade(score: float) -> str:
    if score >= 80: return "A"
    if score >= 60: return "B"
    if score >= 40: return "C"
    if score >= 20: return "D"
    return "E"


def _score_to_color(score: float) -> str:
    if score >= 80: return "#22c55e"
    if score >= 60: return "#84cc16"
    if score >= 40: return "#f59e0b"
    if score >= 20: return "#ef4444"
    return "#7f1d1d"


# ──────────────────────────────────────────────────────────────────────────────
# 5.  DUAL ROUTE
# ──────────────────────────────────────────────────────────────────────────────

def get_dual_routes(
    origin_lat:  float,
    origin_lon:  float,
    dest_lat:    float,
    dest_lon:    float,
    alpha:       float = DEFAULT_ALPHA,
    hour:        int   = DEFAULT_HOUR,
) -> dict:
    G = load_graph()

    # Inject ML scores if missing
    sample_edge = next(iter(G.edges(data=True)))[2]
    if "safety_score" not in sample_edge:
        G = inject_ml_scores(G, hour)

    # Compute time-aware weights
    G = compute_edge_weights(G, alpha=alpha, hour=hour)

    orig_node = find_nearest_node(G, origin_lat, origin_lon)
    dest_node = find_nearest_node(G, dest_lat,   dest_lon)

    log.info(
        f"Routing {origin_lat:.4f},{origin_lon:.4f} → "
        f"{dest_lat:.4f},{dest_lon:.4f} | "
        f"alpha={alpha} hour={hour}"
    )

    safe_nodes = compute_route(G, orig_node, dest_node, "composite_weight")
    fast_nodes = compute_route(G, orig_node, dest_node, "travel_time")

    if safe_nodes is None or fast_nodes is None:
        return {"error": "No route found between these points."}

    safe_stats = compute_route_stats(G, safe_nodes, hour)
    fast_stats = compute_route_stats(G, fast_nodes, hour)

    time_penalty = safe_stats["total_time_min"] - fast_stats["total_time_min"]
    safety_gain  = safe_stats["avg_safety_score"] - fast_stats["avg_safety_score"]

    comparison = {
        "time_penalty_min":    round(time_penalty,  1),
        "safety_gain_points":  round(safety_gain,   1),
        "recommendation": (
            "Take the safer route — minimal time cost."
            if time_penalty <= 5 else
            "Safer route adds significant time. Your choice."
        ),
        "safer_route_worth_it": time_penalty <= 5 or safety_gain >= 15,
    }

    log.info(
        f"Safe route: {safe_stats['avg_safety_score']:.1f} pts, "
        f"{safe_stats['total_time_min']:.1f} min"
    )
    log.info(
        f"Fast route: {fast_stats['avg_safety_score']:.1f} pts, "
        f"{fast_stats['total_time_min']:.1f} min"
    )
    log.info(
        f"Safety gain: +{safety_gain:.1f} pts | "
        f"Time cost: +{time_penalty:.1f} min"
    )

    return {
        "safe_route":  safe_stats,
        "fast_route":  fast_stats,
        "comparison":  comparison,
        "alpha":       alpha,
        "hour":        hour,
        "origin":      {"lat": origin_lat, "lon": origin_lon},
        "destination": {"lat": dest_lat,   "lon": dest_lon},
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6.  TEST
# ──────────────────────────────────────────────────────────────────────────────

def run_test():
    logging.basicConfig(level=logging.INFO)
    log.info("=== routing/dijkstra.py TEST ===")

    test_cases = [
        ("MG Road → Koramangala",    12.9767, 77.6009, 12.9352, 77.6245),
        ("Majestic → Indiranagar",   12.9767, 77.5713, 12.9718, 77.6412),
        ("Shivajinagar → Jayanagar", 12.9839, 77.5929, 12.9220, 77.5833),
    ]

    for name, olat, olon, dlat, dlon in test_cases:
        print(f"\n── {name} ──")
        for hour in [9, 22, 0]:
            result = get_dual_routes(olat, olon, dlat, dlon, alpha=0.7, hour=hour)
            if "error" in result:
                print(f"  {hour:02d}:00 → ERROR: {result['error']}")
                continue
            safe = result["safe_route"]
            fast = result["fast_route"]
            gain = result["comparison"]["safety_gain_points"]
            cost = result["comparison"]["time_penalty_min"]
            print(
                f"  {hour:02d}:00 | "
                f"Safe: {safe['avg_safety_score']:.1f} | "
                f"Fast: {fast['avg_safety_score']:.1f} | "
                f"Gain: {gain:+.1f} | "
                f"Cost: {cost:+.1f} min"
            )

    log.info("=== TEST DONE ===")


if __name__ == "__main__":
    run_test()