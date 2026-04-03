"""
ai/ml/predict.py
================
Loads trained XGBoost model and runs inference on road segments.
Called by the routing engine at request time.
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

from ai.ml.features import (
    FEATURE_COLS,
    FEATURE_DEFAULTS,
    build_feature_vector,
    build_feature_vectors_batch,
    make_time_features,
)

log = logging.getLogger("ml.predict")

ARTIFACTS = Path("ai/ml/artifacts")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_model():
    """
    Loads trained XGBoost model from disk.
    Cached — only loads once per process.
    """
    model_path = ARTIFACTS / "safety_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run training first: python -m ai.ml.train"
        )
    model = joblib.load(model_path)
    log.info(f"Model loaded: {model_path}")
    return model


@lru_cache(maxsize=1)
def load_feature_cols() -> list:
    """Loads feature column list saved during training."""
    path = ARTIFACTS / "feature_cols.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return FEATURE_COLS


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SINGLE SEGMENT PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

def predict_safety_score(
    edge_data: dict,
    hour: int = 22,
) -> float:
    """
    Predicts safety score (0-100) for a single road segment.

    Args:
        edge_data : dict with edge attributes (from OSM graph)
        hour      : hour of day for time-aware scoring

    Returns:
        float safety score 0-100
        (100 = very safe, 0 = very unsafe)
    """
    try:
        model   = load_model()
        feat_df = build_feature_vector(edge_data, hour)
        score   = model.predict(feat_df)[0]
        return float(np.clip(score, 0, 100))
    except Exception as e:
        log.warning(f"Prediction failed: {e}. Using default 40.")
        return 40.0


# ──────────────────────────────────────────────────────────────────────────────
# 3.  BATCH PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

def predict_safety_scores_batch(
    edges: list[dict],
    hour: int = 22,
) -> list[float]:
    """
    Predicts safety scores for multiple segments at once.
    Much faster than calling predict_safety_score() in a loop.

    Args:
        edges : list of edge attribute dicts
        hour  : hour of day

    Returns:
        list of float safety scores 0-100
    """
    if not edges:
        return []

    try:
        model   = load_model()
        feat_df = build_feature_vectors_batch(edges, hour)
        scores  = model.predict(feat_df)
        return [float(np.clip(s, 0, 100)) for s in scores]
    except Exception as e:
        log.warning(f"Batch prediction failed: {e}. Using defaults.")
        return [40.0] * len(edges)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  SCORE WITH SHAP BREAKDOWN
# ──────────────────────────────────────────────────────────────────────────────

def predict_with_explanation(
    edge_data: dict,
    hour: int = 22,
) -> dict:
    """
    Returns safety score AND per-feature SHAP contributions.
    Used by the LLM explainer to generate plain-English explanations.

    Returns dict:
        {
            "safety_score": 67.3,
            "feature_contributions": {
                "luminosity_score": +12.4,
                "crime_penalty":    -18.2,
                ...
            },
            "top_positive": ["luminosity_score", "commercial_score"],
            "top_negative": ["crime_penalty", "darkness_ratio"],
        }
    """
    try:
        from ai.ml.shap_explainer import get_shap_values
        model   = load_model()
        feat_df = build_feature_vector(edge_data, hour)
        score   = float(np.clip(model.predict(feat_df)[0], 0, 100))

        # SHAP contributions
        shap_vals = get_shap_values(feat_df)

        contributions = {}
        for i, col in enumerate(FEATURE_COLS):
            contributions[col] = round(float(shap_vals[0][i]), 3)

        # Sort by absolute value
        sorted_feats = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_positive = [k for k, v in sorted_feats if v > 0][:3]
        top_negative = [k for k, v in sorted_feats if v < 0][:3]

        return {
            "safety_score":          round(score, 2),
            "feature_contributions": contributions,
            "top_positive":          top_positive,
            "top_negative":          top_negative,
            "hour":                  hour,
        }

    except Exception as e:
        log.warning(f"Explanation failed: {e}")
        return {
            "safety_score":          predict_safety_score(edge_data, hour),
            "feature_contributions": {},
            "top_positive":          [],
            "top_negative":          [],
            "hour":                  hour,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  SAFETY GRADE
# ──────────────────────────────────────────────────────────────────────────────

def score_to_grade(score: float) -> str:
    """Converts numeric score to letter grade."""
    if score >= 80: return "A"
    if score >= 60: return "B"
    if score >= 40: return "C"
    if score >= 20: return "D"
    return "E"


def score_to_label(score: float) -> str:
    """Converts numeric score to human-readable label."""
    if score >= 80: return "Very Safe"
    if score >= 60: return "Safe"
    if score >= 40: return "Moderate"
    if score >= 20: return "Unsafe"
    return "Avoid"


def score_to_color(score: float) -> str:
    """Converts score to hex color for map visualization."""
    if score >= 80: return "#22c55e"   # green
    if score >= 60: return "#84cc16"   # light green
    if score >= 40: return "#f59e0b"   # amber
    if score >= 20: return "#ef4444"   # red
    return "#7f1d1d"                   # dark red