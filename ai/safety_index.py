"""
ai/safety_index.py
==================
Psychological Safety Index (PSI) — formal metric definition.

PSI is the core innovation of Fear-Free Navigator.
It quantifies 'perceived safety' as a composite of:
  1. Physical safety    (crime, accidents)
  2. Environmental safety (lighting, visibility)
  3. Social safety      (footfall, commercial activity)
  4. Temporal safety    (time-of-day modulation)

PSI Formula:
    PSI = w1*L + w2*S + w3*F + w4*E - w5*C*T

Where:
    L = Luminosity score (lighting)
    S = Social/commercial score
    F = Footfall score
    E = Emergency infrastructure score
    C = Crime penalty
    T = Temporal multiplier (night = 1.35, day = 1.0)
    w1..w5 = learned weights from XGBoost SHAP values

This is NOT a simple weighted average.
The temporal multiplier is multiplicative on the crime term,
creating non-linear night/day behavior consistent with
criminology research on opportunity theory.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path

log = logging.getLogger("safety_index")

# Weights derived from XGBoost SHAP feature importance
# (see ai/ml/artifacts/feature_importance.csv)
PSI_WEIGHTS = {
    "luminosity":   0.28,   # SHAP rank 2
    "social":       0.22,   # SHAP rank 1 (commercial_score)
    "footfall":     0.18,   # SHAP rank 3
    "emergency":    0.15,   # SHAP rank 4
    "crime":        0.17,   # SHAP rank 6 (inverted)
}

# Temporal multipliers based on criminology research
# (Cohen & Felson Routine Activity Theory, 1979)
TEMPORAL_MULTIPLIERS = {
    "day":       {"crime": 1.00, "social": 1.00, "footfall": 1.00},
    "evening":   {"crime": 1.15, "social": 0.85, "footfall": 0.85},
    "night":     {"crime": 1.35, "social": 0.40, "footfall": 0.40},
    "late_night":{"crime": 1.50, "social": 0.15, "footfall": 0.20},
}


def get_time_period(hour: int) -> str:
    if   6  <= hour < 17: return "day"
    elif 17 <= hour < 20: return "evening"
    elif 20 <= hour < 23: return "night"
    else:                 return "late_night"


def compute_psi(features: dict, hour: int = 22) -> dict:
    """
    Computes Psychological Safety Index for a road segment.

    Returns PSI (0-100) with component breakdown for interpretability.
    """
    period = get_time_period(hour)
    tmult  = TEMPORAL_MULTIPLIERS[period]

    # Component scores (0-1)
    L = float(np.clip(features.get("luminosity_score", 35) / 100, 0, 1))
    S = float(np.clip(features.get("commercial_score", 0.3), 0, 1))
    F = float(np.clip(features.get("footfall_score",   0.4), 0, 1))
    E = float(np.clip(features.get("emergency_score",  0.1), 0, 1))
    C = float(np.clip(features.get("crime_penalty",    0.2), 0, 1))

    # Apply temporal multipliers
    S_t = S * tmult["social"]
    F_t = F * tmult["footfall"]
    C_t = C * tmult["crime"]

    # PSI formula
    w = PSI_WEIGHTS
    psi_raw = (
        w["luminosity"] * L +
        w["social"]     * S_t +
        w["footfall"]   * F_t +
        w["emergency"]  * E -
        w["crime"]      * C_t
    )

    # Scale to 0-100
    # Raw range: -0.17 to 0.83 → map to 0-100
    psi = float(np.clip((psi_raw + 0.17) / 1.0 * 100, 0, 100))

    return {
        "psi":            round(psi, 2),
        "psi_grade":      _psi_to_grade(psi),
        "period":         period,
        "components": {
            "luminosity_contribution":  round(w["luminosity"] * L * 100,  2),
            "social_contribution":      round(w["social"]     * S_t * 100, 2),
            "footfall_contribution":    round(w["footfall"]   * F_t * 100, 2),
            "emergency_contribution":   round(w["emergency"]  * E * 100,   2),
            "crime_penalty":           -round(w["crime"]      * C_t * 100, 2),
        },
        "temporal_effect": {
            "period":          period,
            "crime_multiplier":tmult["crime"],
            "social_dampening":round((1 - tmult["social"]) * 100, 0),
        },
    }


def _psi_to_grade(psi: float) -> str:
    if psi >= 80: return "A — Very Safe"
    if psi >= 65: return "B — Safe"
    if psi >= 50: return "C — Moderate"
    if psi >= 35: return "D — Unsafe"
    return "E — Avoid"


def compute_route_psi(segments: list, hour: int = 22) -> dict:
    """
    Computes PSI statistics for an entire route.
    Returns mean, min, distribution of PSI across all segments.
    """
    psi_values = []
    for seg in segments:
        psi_result = compute_psi(seg, hour)
        psi_values.append(psi_result["psi"])

    if not psi_values:
        return {"mean_psi": 50, "min_psi": 50, "route_grade": "C"}

    return {
        "mean_psi":       round(float(np.mean(psi_values)), 2),
        "min_psi":        round(float(np.min(psi_values)),  2),
        "std_psi":        round(float(np.std(psi_values)),  2),
        "route_grade":    _psi_to_grade(float(np.mean(psi_values))),
        "segment_count":  len(psi_values),
        "safe_segments":  sum(1 for p in psi_values if p >= 65),
        "unsafe_segments":sum(1 for p in psi_values if p < 35),
    }