"""
routing/graph.py
================
Graph loading and management utilities.
Handles loading, caching and updating the scored road graph.
"""

import logging
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
from pathlib import Path
from functools import lru_cache

log = logging.getLogger("routing.graph")

DATA_PROCESSED = Path("data/processed")
DATA_RAW       = Path("data/raw")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD GRAPH
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_graph():
    """
    Loads road graph with safety scores injected.
    Cached — loads once per process lifetime.
    Priority:
        1. bengaluru_scored_graph.graphml  (has safety scores)
        2. bengaluru_graph.graphml         (base graph, scores injected)
    """
    scored = DATA_PROCESSED / "bengaluru_scored_graph.graphml"
    base   = DATA_RAW       / "bengaluru_graph.graphml"

    if scored.exists():
        log.info(f"Loading scored graph → {scored}")
        G = ox.load_graphml(scored)
        log.info(f"Scored graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
        return G

    if base.exists():
        log.info(f"Loading base graph → {base}")
        G = ox.load_graphml(base)
        G = inject_scores_from_csv(G)
        log.info(f"Base graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
        return G

    raise FileNotFoundError(
        "No graph found.\n"
        "Run: python -m ingestion.fetch_osm"
    )


def inject_scores_from_csv(G) -> nx.MultiDiGraph:
    """
    Reads ML-predicted scores from CSV and injects into graph edges.
    Used when scored graphml is not available.
    """
    ml_path = DATA_PROCESSED / "bengaluru_feature_store_ml.csv"
    if not ml_path.exists():
        log.warning("ML scores CSV not found. Using default score 40.")
        for u, v, k, data in G.edges(data=True, keys=True):
            data["safety_score"] = 40.0
        return G

    log.info("Injecting ML scores from CSV ...")
    df = pd.read_csv(ml_path)

    lookup = {}
    for _, row in df.iterrows():
        key = (str(row["u"]), str(row["v"]), str(row["key"]))
        lookup[key] = {
            "safety_score":    float(row.get("safety_score_ml",    40.0)),
            "crime_density":   float(row.get("crime_density",       0.2)),
            "luminosity_score":float(row.get("luminosity_score",   35.0)),
        }

    injected = 0
    missing  = 0
    for u, v, k, data in G.edges(data=True, keys=True):
        key    = (str(u), str(v), str(k))
        scores = lookup.get(key)
        if scores:
            data.update(scores)
            injected += 1
        else:
            data["safety_score"]     = 40.0
            data["crime_density"]    = 0.2
            data["luminosity_score"] = 35.0
            missing += 1

    log.info(f"Injected: {injected:,} | Default: {missing:,}")
    return G


# ──────────────────────────────────────────────────────────────────────────────
# 2.  GRAPH STATS
# ──────────────────────────────────────────────────────────────────────────────

def get_graph_stats(G) -> dict:
    """Returns summary statistics about the graph."""
    scores = [
        data.get("safety_score", 40.0)
        for _, _, data in G.edges(data=True)
    ]
    times = [
        data.get("travel_time", 60.0)
        for _, _, data in G.edges(data=True)
    ]

    return {
        "n_nodes":          len(G.nodes),
        "n_edges":          len(G.edges),
        "avg_safety_score": round(float(np.mean(scores)), 2),
        "min_safety_score": round(float(np.min(scores)),  2),
        "max_safety_score": round(float(np.max(scores)),  2),
        "avg_travel_time_s":round(float(np.mean(times)),  2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  HEATMAP DATA
# ──────────────────────────────────────────────────────────────────────────────

def get_heatmap_data(G, sample_n: int = 5000) -> list:
    """
    Returns safety score heatmap data for frontend visualization.
    Samples edges for performance — returns list of
    [lat, lon, safety_score] for each sampled edge midpoint.
    """
    records = []

    for u, v, data in G.edges(data=True):
        u_lat = G.nodes[u]["y"]
        u_lon = G.nodes[u]["x"]
        v_lat = G.nodes[v]["y"]
        v_lon = G.nodes[v]["x"]

        mid_lat = (u_lat + v_lat) / 2
        mid_lon = (u_lon + v_lon) / 2
        score   = float(data.get("safety_score", 40.0))

        records.append([
            round(mid_lat, 6),
            round(mid_lon, 6),
            round(score,   1),
        ])

    # Sample for performance
    if len(records) > sample_n:
        import random
        random.seed(42)
        records = random.sample(records, sample_n)

    return records


# ──────────────────────────────────────────────────────────────────────────────
# 4.  NEAREST NODE
# ──────────────────────────────────────────────────────────────────────────────

def nearest_node(G, lat: float, lon: float) -> int:
    """Returns nearest graph node ID to given coordinates."""
    return ox.nearest_nodes(G, lon, lat)