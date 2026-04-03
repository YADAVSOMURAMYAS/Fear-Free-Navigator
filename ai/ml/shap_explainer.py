"""
ai/ml/shap_explainer.py
========================
SHAP-based explainability for the safety score model.
Provides per-feature contribution values for each road segment.
Used by the LLM explainer to generate plain-English explanations.
"""

import logging
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from functools import lru_cache

log = logging.getLogger("ml.shap")

ARTIFACTS = Path("ai/ml/artifacts")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD EXPLAINER
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_explainer() -> shap.TreeExplainer:
    """
    Loads pre-computed SHAP TreeExplainer from disk.
    Cached — only loads once per process.
    """
    path = ARTIFACTS / "shap_explainer.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"SHAP explainer not found: {path}\n"
            "Run training first: python -m ai.ml.train"
        )
    explainer = joblib.load(path)
    log.info(f"SHAP explainer loaded: {path}")
    return explainer


# ──────────────────────────────────────────────────────────────────────────────
# 2.  GET SHAP VALUES
# ──────────────────────────────────────────────────────────────────────────────

def get_shap_values(feat_df: pd.DataFrame) -> np.ndarray:
    """
    Returns SHAP values for a feature DataFrame.

    Args:
        feat_df : DataFrame with FEATURE_COLS columns

    Returns:
        np.ndarray of shape (n_samples, n_features)
    """
    explainer  = load_explainer()
    shap_vals  = explainer.shap_values(feat_df)
    return shap_vals


def get_feature_contributions(
    feat_df: pd.DataFrame,
    feature_cols: list,
) -> dict:
    """
    Returns dict mapping feature name → SHAP contribution value.
    Positive = increases safety score.
    Negative = decreases safety score.
    """
    shap_vals = get_shap_values(feat_df)

    if len(shap_vals.shape) == 1:
        vals = shap_vals
    else:
        vals = shap_vals[0]

    return {
        col: round(float(vals[i]), 4)
        for i, col in enumerate(feature_cols)
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  HUMAN-READABLE EXPLANATION
# ──────────────────────────────────────────────────────────────────────────────

def build_explanation_context(
    contributions: dict,
    safety_score: float,
    hour: int = 22,
) -> str:
    """
    Builds a structured text summary of why a segment got its score.
    This is fed as context to the LLM explainer.
    """
    from ai.ml.features import get_feature_description

    # Sort by absolute SHAP value
    sorted_items = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    positives = [(k, v) for k, v in sorted_items if v > 0.5][:4]
    negatives = [(k, v) for k, v in sorted_items if v < -0.5][:4]

    time_str  = f"{hour:02d}:00"
    period    = "night" if (hour >= 20 or hour <= 5) else "day"

    lines = [
        f"Road segment safety analysis at {time_str} ({period})",
        f"Safety score: {safety_score:.1f}/100",
        "",
        "Positive factors (increasing safety):",
    ]
    if positives:
        for feat, val in positives:
            desc = get_feature_description(feat)
            lines.append(f"  + {desc}: +{val:.1f} pts")
    else:
        lines.append("  None significant")

    lines.append("")
    lines.append("Negative factors (decreasing safety):")
    if negatives:
        for feat, val in negatives:
            desc = get_feature_description(feat)
            lines.append(f"  - {desc}: {val:.1f} pts")
    else:
        lines.append("  None significant")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  QUICK TEST
# ──────────────────────────────────────────────────────────────────────────────

def test_explainer():
    """Quick smoke test — run after training."""
    from ai.ml.features import FEATURE_COLS, FEATURE_DEFAULTS

    log.info("Testing SHAP explainer ...")
    test_row = pd.DataFrame([FEATURE_DEFAULTS])[FEATURE_COLS].fillna(0)

    contribs = get_feature_contributions(test_row, FEATURE_COLS)
    context  = build_explanation_context(contribs, safety_score=55.0, hour=22)

    log.info("SHAP test output:")
    log.info(context)
    log.info("SHAP explainer test PASSED.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_explainer()