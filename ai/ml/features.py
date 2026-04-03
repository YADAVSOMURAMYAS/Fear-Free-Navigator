"""
ai/ml/features.py
=================
Feature engineering utilities.
Used by train.py, predict.py and the FastAPI routing engine.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Feature columns (single source of truth) ──────────────────────────────────
FEATURE_COLS = [
    # Category 1 — Illumination
    "luminosity_score",
    "luminosity_norm",
    "lamp_count_80m_norm",
    "lit_road_bonus",

    # Category 2 — Commercial
    "commercial_score",
    "emergency_score",
    "shop_count_200m_norm",
    "police_count_500m_norm",

    # Category 3 — Footfall
    "footfall_score",
    "transit_score",
    "bus_stop_count_300m_norm",
    "is_primary_secondary",
    "has_sidewalk",

    # Category 4 — Crime
    "crime_penalty",
    "crime_density",
    "night_crime_density",
    "accident_density",
    "combined_risk_score",

    # Category 5 — Physical
    "physical_score",
    "is_dead_end",
    "highway_type_enc",
    "has_road_name",
    "lanes",
    "cctv_count_150m_norm",
    "construction_nearby",

    # Category 6 — Visual
    "visual_score",
    "brightness_mean",
    "darkness_ratio",
    "greenery_ratio",

    # Time
    "hour_sin",
    "hour_cos",
    "is_night",
]

# ── Feature defaults (used when a value is missing) ────────────────────────────
FEATURE_DEFAULTS = {
    "luminosity_score":       35.0,
    "luminosity_norm":         0.35,
    "lamp_count_80m_norm":     0.0,
    "lit_road_bonus":          0.3,
    "commercial_score":        0.3,
    "emergency_score":         0.1,
    "shop_count_200m_norm":    0.2,
    "police_count_500m_norm":  0.0,
    "footfall_score":          0.4,
    "transit_score":           0.2,
    "bus_stop_count_300m_norm":0.1,
    "is_primary_secondary":    0,
    "has_sidewalk":            0,
    "crime_penalty":           0.2,
    "crime_density":           0.15,
    "night_crime_density":     0.20,
    "accident_density":        0.10,
    "combined_risk_score":     20.0,
    "physical_score":          0.5,
    "is_dead_end":             0,
    "highway_type_enc":        0.42,
    "has_road_name":           1,
    "lanes":                   1,
    "cctv_count_150m_norm":    0.0,
    "construction_nearby":     0,
    "visual_score":            0.5,
    "brightness_mean":         0.35,
    "darkness_ratio":          0.30,
    "greenery_ratio":          0.10,
    "hour_sin":                0.0,
    "hour_cos":               -1.0,   # default hour=22 (10PM)
    "is_night":                1,
}


def make_time_features(hour: int) -> dict:
    """
    Creates cyclical time features from hour of day.
    Cyclical encoding ensures hour 23 and hour 0 are close together.
    """
    return {
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "is_night": int(hour >= 20 or hour <= 5),
    }


def get_time_period(hour: int) -> str:
    """Returns time period label for logging/display."""
    if   6  <= hour < 20: return "day"
    elif 20 <= hour < 23: return "evening"
    elif hour == 23 or hour < 3: return "night"
    else:                 return "late_night"


def build_feature_vector(edge_data: dict, hour: int = 22) -> pd.DataFrame:
    """
    Builds a single-row feature DataFrame for one road segment.
    Used by the routing engine at inference time.

    Args:
        edge_data : dict of raw edge attributes from OSM graph
        hour      : hour of day (0-23) for time-aware scoring

    Returns:
        Single-row DataFrame with all FEATURE_COLS
    """
    time_feats = make_time_features(hour)

    row = {}
    for col in FEATURE_COLS:
        if col in edge_data:
            row[col] = edge_data[col]
        elif col in time_feats:
            row[col] = time_feats[col]
        else:
            row[col] = FEATURE_DEFAULTS.get(col, 0.0)

    return pd.DataFrame([row])[FEATURE_COLS]


def build_feature_vectors_batch(
    edges: list[dict],
    hour: int = 22,
) -> pd.DataFrame:
    """
    Builds feature DataFrame for multiple road segments at once.
    More efficient than calling build_feature_vector() in a loop.

    Args:
        edges : list of edge attribute dicts
        hour  : hour of day

    Returns:
        DataFrame with len(edges) rows and all FEATURE_COLS
    """
    time_feats = make_time_features(hour)
    records    = []

    for edge_data in edges:
        row = {}
        for col in FEATURE_COLS:
            if col in edge_data:
                row[col] = edge_data[col]
            elif col in time_feats:
                row[col] = time_feats[col]
            else:
                row[col] = FEATURE_DEFAULTS.get(col, 0.0)
        records.append(row)

    return pd.DataFrame(records)[FEATURE_COLS].fillna(0)


def normalise_count(value: float, max_val: float) -> float:
    """Normalises a count feature to 0-1 range."""
    return float(np.clip(value / max_val, 0.0, 1.0))


def get_feature_description(feature_name: str) -> str:
    """
    Returns human-readable description of a feature.
    Used by the LLM explainer to build prompts.
    """
    descriptions = {
        "luminosity_score":        "nighttime lighting brightness (satellite)",
        "luminosity_norm":         "normalised luminosity 0-1",
        "lamp_count_80m_norm":     "street lamps within 80m",
        "lit_road_bonus":          "road tagged as lit in OSM",
        "commercial_score":        "nearby shops and restaurants",
        "emergency_score":         "nearby police stations and hospitals",
        "shop_count_200m_norm":    "shops within 200m",
        "police_count_500m_norm":  "police stations within 500m",
        "footfall_score":          "estimated pedestrian footfall",
        "transit_score":           "public transport accessibility",
        "bus_stop_count_300m_norm":"bus stops within 300m",
        "is_primary_secondary":    "major road (high traffic)",
        "has_sidewalk":            "dedicated footpath present",
        "crime_penalty":           "crime risk score (higher = more dangerous)",
        "crime_density":           "historical crime density nearby",
        "night_crime_density":     "night-time crime density nearby",
        "accident_density":        "road accident hotspot proximity",
        "combined_risk_score":     "combined crime and accident risk",
        "physical_score":          "physical road environment quality",
        "is_dead_end":             "isolated dead-end road",
        "highway_type_enc":        "road hierarchy (primary=safer, path=unsafe)",
        "has_road_name":           "named road (more recognised/used)",
        "lanes":                   "number of lanes (more = busier)",
        "cctv_count_150m_norm":    "CCTV cameras within 150m",
        "construction_nearby":     "active construction zone nearby",
        "visual_score":            "visual safety from street imagery",
        "brightness_mean":         "mean image brightness",
        "darkness_ratio":          "fraction of very dark pixels",
        "greenery_ratio":          "vegetation / greenery visible",
        "hour_sin":                "time of day (sine encoding)",
        "hour_cos":                "time of day (cosine encoding)",
        "is_night":                "night-time flag (8PM-6AM)",
    }
    return descriptions.get(feature_name, feature_name.replace("_", " "))