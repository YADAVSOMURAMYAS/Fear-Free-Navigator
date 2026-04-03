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

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PROCESSED = Path("data/processed")
ARTIFACTS      = Path("ai/ml/artifacts")

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_ALPHA      = 0.7     # safety weight
MIN_SAFETY_SCORE   = 1.0     # avoid division by zero
DEFAULT_HOUR       = 22      # default routing hour (10 PM)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH LOADING
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_graph():
    """
    Loads the scored road graph.
    Tries ML-scored graph first, falls back to base graph.
    Cached — loads once per process.
    """
    # Try scored graph first
    scored_path = DATA_PROCESSED / "bengaluru_scored_graph.graphml"
    base_path   = Path("data/raw/bengaluru_graph.graphml")

    if scored_path.exists():
        log.info(f"Loading scored graph: {scored_path}")
        G = ox.load_graphml(scored_path)
    elif base_path.exists():
        log.info(f"Loading base graph: {base_path}")
        G = ox.load_graphml(base_path)
    else:
        raise FileNotFoundError(
            "No graph found. Run ingestion pipeline first."
        )

    log.info(f"Graph loaded: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
    return G


def inject_ml_scores(G, hour: int = 22):
    """
    Loads ML-predicted safety scores from feature store
    and injects them into graph edges.
    Called when scored graph is not available.
    """
    ml_path = DATA_PROCESSED / "bengaluru_feature_store_ml.csv"
    if not ml_path.exists():
        log.warning("ML scores not found. Using preliminary scores.")
        return G

    log.info("Injecting ML safety scores into graph edges ...")
    df = pd.read_csv(ml_path)

    # Build lookup
    score_lookup = {}
    for _, row in df.iterrows():
        key = (str(row["u"]), str(row["v"]), str(row["key"]))
        score_lookup[key] = float(row.get("safety_score_ml", 40.0))

    injected = 0
    for u, v, k, data in G.edges(data=True, keys=True):
        lookup_key = (str(u), str(v), str(k))
        score = score_lookup.get(lookup_key, 40.0)
        data["safety_score"] = score
        injected += 1

    log.info(f"Injected scores into {injected:,} edges")
    return G


# ──────────────────────────────────────────────────────────────────────────────
# 2.  WEIGHT COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_edge_weights(G, alpha: float = DEFAULT_ALPHA, hour: int = DEFAULT_HOUR):
    """
    Computes composite edge weights for routing.

    Formula:
        weight = alpha * (100/safety_score) + (1-alpha) * travel_time_norm

    Higher weight = edge avoided (Dijkstra finds minimum weight path).
    Low safety score → high weight → avoided when alpha > 0.
    High travel time → high weight → avoided when alpha < 1.
    """
    # Get travel time range for normalisation
    times = [
        data.get("travel_time", 60)
        for _, _, data in G.edges(data=True)
    ]
    max_time = max(times) if times else 300
    min_time = min(times) if times else 1

    for u, v, key, data in G.edges(data=True, keys=True):
        safety = float(data.get("safety_score", 40.0))
        safety = max(safety, MIN_SAFETY_SCORE)

        travel_time = float(data.get("travel_time", 60.0))

        # Normalise travel time to 0-100 scale
        time_norm = (travel_time - min_time) / (max_time - min_time + 1) * 100

        # Composite weight
        # alpha=1 → only safety matters
        # alpha=0 → only time matters
        data["composite_weight"] = (
            alpha * (100.0 / safety) +
            (1 - alpha) * time_norm
        )

    return G


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ROUTE COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def find_nearest_node(G, lat: float, lon: float) -> int:
    """Finds the nearest graph node to a lat/lon coordinate."""
    return ox.nearest_nodes(G, lon, lat)


def compute_route(
    G,
    orig_node: int,
    dest_node: int,
    weight: str = "composite_weight",
) -> Optional[list]:
    """
    Runs Dijkstra on the graph.
    Returns list of node IDs or None if no path found.
    """
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
    """
    Computes statistics for a route.
    Returns dict with safety scores, travel time, coordinates.
    """
    if not route or len(route) < 2:
        return {}

    edge_scores    = []
    edge_times     = []
    edge_lengths   = []
    segments       = []
    coords         = []

    for node in route:
        coords.append([
            G.nodes[node]["y"],
            G.nodes[node]["x"],
        ])

    for u, v in zip(route, route[1:]):
        data = G[u][v][0]

        score  = float(data.get("safety_score",  40.0))
        ttime  = float(data.get("travel_time",   60.0))
        length = float(data.get("length",        50.0))
        hw     = data.get("highway", "unknown")
        if isinstance(hw, list):
            hw = hw[0]

        edge_scores.append(score)
        edge_times.append(ttime)
        edge_lengths.append(length)

        segments.append({
            "u":              u,
            "v":              v,
            "safety_score":   round(score, 1),
            "travel_time_s":  round(ttime, 1),
            "length_m":       round(length, 1),
            "highway":        hw,
            "name":           str(data.get("name", "") or ""),
            "safety_grade":   _score_to_grade(score),
            "safety_color":   _score_to_color(score),
        })

    total_time_min = sum(edge_times) / 60
    total_dist_km  = sum(edge_lengths) / 1000
    avg_safety     = float(np.mean(edge_scores))
    min_safety     = float(np.min(edge_scores))

    # Find dangerous segments (score < 30)
    dangerous = [
        s for s in segments if s["safety_score"] < 30
    ]

    return {
        "coords":           coords,
        "segments":         segments,
        "avg_safety_score": round(avg_safety, 1),
        "min_safety_score": round(min_safety, 1),
        "total_time_min":   round(total_time_min, 1),
        "total_dist_km":    round(total_dist_km, 2),
        "n_segments":       len(segments),
        "dangerous_count":  len(dangerous),
        "hour":             hour,
        "safety_grade":     _score_to_grade(avg_safety),
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
# 5.  DUAL ROUTE (SAFE + FAST)
# ──────────────────────────────────────────────────────────────────────────────

def get_dual_routes(
    origin_lat:  float,
    origin_lon:  float,
    dest_lat:    float,
    dest_lon:    float,
    alpha:       float = DEFAULT_ALPHA,
    hour:        int   = DEFAULT_HOUR,
) -> dict:
    """
    Main routing function — returns BOTH safe and fast routes.

    Args:
        origin_lat, origin_lon : start coordinates
        dest_lat,   dest_lon   : end coordinates
        alpha                  : safety weight 0-1
        hour                   : hour of day for time-aware scoring

    Returns dict with:
        safe_route : safety-optimized route stats + coords
        fast_route : time-optimized route stats + coords
        comparison : side-by-side comparison metrics
    """
    G = load_graph()

    # Inject ML scores if not already present
    sample_edge = next(iter(G.edges(data=True)))[2]
    if "safety_score" not in sample_edge:
        G = inject_ml_scores(G, hour)

    # Compute composite weights
    G = compute_edge_weights(G, alpha=alpha, hour=hour)

    # Find nearest nodes
    orig_node = find_nearest_node(G, origin_lat, origin_lon)
    dest_node = find_nearest_node(G, dest_lat,   dest_lon)

    log.info(
        f"Routing {origin_lat:.4f},{origin_lon:.4f} → "
        f"{dest_lat:.4f},{dest_lon:.4f} | "
        f"alpha={alpha} hour={hour}"
    )

    # Safe route — composite weight
    safe_nodes = compute_route(G, orig_node, dest_node, "composite_weight")

    # Fast route — travel time only
    fast_nodes = compute_route(G, orig_node, dest_node, "travel_time")

    if safe_nodes is None or fast_nodes is None:
        return {"error": "No route found between these points."}

    safe_stats = compute_route_stats(G, safe_nodes, hour)
    fast_stats = compute_route_stats(G, fast_nodes, hour)

    # Comparison metrics
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
        "safe_route": safe_stats,
        "fast_route": fast_stats,
        "comparison": comparison,
        "alpha":      alpha,
        "hour":       hour,
        "origin":     {"lat": origin_lat, "lon": origin_lon},
        "destination":{"lat": dest_lat,   "lon": dest_lon},
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6.  TEST
# ──────────────────────────────────────────────────────────────────────────────

def run_test():
    """
    Tests routing between known Bengaluru landmarks.
    MG Road → Koramangala
    """
    logging.basicConfig(level=logging.INFO)
    log.info("=== routing/dijkstra.py TEST ===")

    # MG Road → Koramangala
    result = get_dual_routes(
        origin_lat  = 12.9767,
        origin_lon  = 77.6009,
        dest_lat    = 12.9352,
        dest_lon    = 77.6245,
        alpha       = 0.7,
        hour        = 22,
    )

    if "error" in result:
        log.error(f"Routing failed: {result['error']}")
        return

    print("\n" + "="*55)
    print("ROUTING TEST RESULTS")
    print("="*55)
    print(f"SAFE  route: {result['safe_route']['avg_safety_score']:.1f} safety | "
          f"{result['safe_route']['total_time_min']:.1f} min | "
          f"{result['safe_route']['total_dist_km']:.1f} km")
    print(f"FAST  route: {result['fast_route']['avg_safety_score']:.1f} safety | "
          f"{result['fast_route']['total_time_min']:.1f} min | "
          f"{result['fast_route']['total_dist_km']:.1f} km")
    print(f"Safety gain : +{result['comparison']['safety_gain_points']:.1f} pts")
    print(f"Time cost   : +{result['comparison']['time_penalty_min']:.1f} min")
    print(f"Verdict     : {result['comparison']['recommendation']}")
    print("="*55)

    log.info("=== TEST DONE ===")
    return result


if __name__ == "__main__":
    run_test()