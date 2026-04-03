"""
ai/ml/train.py
==============
Trains XGBoost safety score model on Bengaluru feature store.
Runs on CPU locally — finishes in 3-5 minutes.

Run:
    python -m ai.ml.train

Output in ai/ml/artifacts/:
    safety_model.pkl
    shap_explainer.pkl
    eval_metrics.json
    shap_summary.png
    feature_importance.csv
    prediction_vs_actual.png
"""

import os
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("ml.train")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PROCESSED = Path("data/processed")
ARTIFACTS      = Path("ai/ml/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ── Feature columns ────────────────────────────────────────────────────────────
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

    # Category 6 — Visual / CV
    "visual_score",
    "brightness_mean",
    "darkness_ratio",
    "greenery_ratio",

    # Time features
    "hour_sin",
    "hour_cos",
    "is_night",
]

TARGET_COL = "safety_score"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_feature_store() -> pd.DataFrame:
    path = DATA_PROCESSED / "bengaluru_feature_store.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature store not found: {path}\n"
            "Run build_feature_store.py first:\n"
            "  python -m ingestion.build_feature_store"
        )
    df = pd.read_csv(path)
    log.info(f"Feature store loaded: {df.shape}")
    log.info(f"Safety score stats:\n{df[TARGET_COL].describe()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PREPARE FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> tuple:
    # Add missing columns with defaults
    for col in FEATURE_COLS:
        if col not in df.columns:
            log.warning(f"Missing feature '{col}' — using default 0")
            df[col] = 0.0

    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL].fillna(50)

    # Remove edge-case rows
    mask = (y > 0.5) & (y < 99.5)
    X    = X[mask]
    y    = y[mask]
    log.info(f"Samples after filter: {len(X):,}")
    log.info(f"Target range: {y.min():.1f} – {y.max():.1f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log.info(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────────────────
# 3.  TRAIN MODEL
# ──────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, X_test, y_train, y_test) -> xgb.XGBRegressor:
    log.info("Training XGBoost model on CPU ...")

    model = xgb.XGBRegressor(
        n_estimators          = 800,
        max_depth             = 6,
        min_child_weight      = 5,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        reg_alpha             = 0.1,
        reg_lambda            = 1.5,
        gamma                 = 0.1,
        objective             = "reg:squarederror",
        random_state          = 42,
        n_jobs                = -1,
        tree_method           = "hist",
        early_stopping_rounds = 50,
    )

    model.fit(
        X_train, y_train,
        eval_set = [(X_train, y_train), (X_test, y_test)],
        verbose  = 100,
    )

    log.info(f"Best iteration: {model.best_iteration}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 4.  EVALUATE
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, X_train, y_train) -> dict:
    y_pred       = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    mae      = mean_absolute_error(y_test, y_pred)
    rmse     = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2       = r2_score(y_test, y_pred)
    mae_train= mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    log.info("=" * 55)
    log.info("MODEL EVALUATION RESULTS")
    log.info("=" * 55)
    log.info(f"  Test  MAE  : {mae:.3f}  (avg error in safety points)")
    log.info(f"  Test  RMSE : {rmse:.3f}")
    log.info(f"  Test  R²   : {r2:.4f}  (1.0 = perfect)")
    log.info(f"  Train MAE  : {mae_train:.3f}")
    log.info(f"  Train R²   : {r2_train:.4f}")
    log.info("=" * 55)

    # Prediction vs actual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=5, color="steelblue")
    plt.plot([0, 100], [0, 100], "r--", linewidth=2, label="Perfect")
    plt.xlabel("Actual Safety Score")
    plt.ylabel("Predicted Safety Score")
    plt.title("XGBoost: Predicted vs Actual Safety Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "prediction_vs_actual.png", dpi=150)
    plt.close()
    log.info(f"Plot saved → {ARTIFACTS}/prediction_vs_actual.png")

    return {
        "mae":            round(float(mae),      4),
        "rmse":           round(float(rmse),     4),
        "r2":             round(float(r2),       4),
        "mae_train":      round(float(mae_train),4),
        "r2_train":       round(float(r2_train), 4),
        "n_train":        int(len(X_train)),
        "n_test":         int(len(X_test)),
        "best_iteration": int(model.best_iteration),
        "n_features":     int(len(FEATURE_COLS)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  SHAP EXPLAINABILITY
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap(model, X_test: pd.DataFrame) -> shap.TreeExplainer:
    log.info("Computing SHAP values ...")

    explainer   = shap.TreeExplainer(model)
    sample_size = min(500, len(X_test))
    shap_values = explainer.shap_values(X_test.iloc[:sample_size])

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test.iloc[:sample_size],
        feature_names=FEATURE_COLS,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        ARTIFACTS / "shap_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    log.info(f"SHAP plot → {ARTIFACTS}/shap_summary.png")

    # Feature importance from SHAP
    importance = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": np.abs(shap_values).mean(0),
    }).sort_values("importance", ascending=False)

    importance.to_csv(ARTIFACTS / "feature_importance.csv", index=False)

    log.info("\nTop 10 most important features:")
    log.info(importance.head(10).to_string(index=False))

    return explainer


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PREDICT ALL SEGMENTS
# ──────────────────────────────────────────────────────────────────────────────

def predict_all_segments(model, df: pd.DataFrame) -> pd.DataFrame:
    """Runs model on all segments and saves ML-scored feature store."""
    log.info("Predicting safety scores for all segments ...")

    X_all     = df[FEATURE_COLS].fillna(0)
    all_preds = model.predict(X_all)

    df["safety_score_ml"] = np.clip(all_preds, 0, 100).round(2)
    df["safety_grade_ml"] = pd.cut(
        df["safety_score_ml"],
        bins   = [0,  20,  40,  60,  80, 100],
        labels = ["E","D", "C", "B", "A"],
        include_lowest=True,
    )

    out = DATA_PROCESSED / "bengaluru_feature_store_ml.csv"
    df.to_csv(out, index=False)
    log.info(f"ML-scored feature store → {out}")

    log.info("\nML Safety Score Distribution:")
    log.info(df["safety_grade_ml"].value_counts().sort_index().to_string())
    log.info(f"Mean ML score: {df['safety_score_ml'].mean():.1f}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 7.  SAVE ARTIFACTS
# ──────────────────────────────────────────────────────────────────────────────

def save_artifacts(model, explainer, metrics: dict) -> None:
    # Model
    joblib.dump(model,     ARTIFACTS / "safety_model.pkl")
    joblib.dump(explainer, ARTIFACTS / "shap_explainer.pkl")

    # Feature list — must match exactly at inference time
    with open(ARTIFACTS / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    # Metrics
    with open(ARTIFACTS / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("=" * 55)
    log.info("ALL ARTIFACTS SAVED")
    log.info("=" * 55)
    log.info(f"  Model      : {ARTIFACTS}/safety_model.pkl")
    log.info(f"  Explainer  : {ARTIFACTS}/shap_explainer.pkl")
    log.info(f"  Features   : {ARTIFACTS}/feature_cols.json")
    log.info(f"  Metrics    : {ARTIFACTS}/eval_metrics.json")
    log.info(f"  SHAP plot  : {ARTIFACTS}/shap_summary.png")
    log.info(f"  Pred plot  : {ARTIFACTS}/prediction_vs_actual.png")
    log.info(f"  Importance : {ARTIFACTS}/feature_importance.csv")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run():
    log.info("=== ai/ml/train.py  START ===")

    # Step 1 — Load feature store
    df = load_feature_store()

    # Step 2 — Prepare features
    X_train, X_test, y_train, y_test = prepare_features(df)

    # Step 3 — Train XGBoost
    model = train_xgboost(X_train, X_test, y_train, y_test)

    # Step 4 — Evaluate
    metrics = evaluate_model(model, X_test, y_test, X_train, y_train)

    # Step 5 — SHAP explainability
    explainer = compute_shap(model, X_test)

    # Step 6 — Predict all segments
    predict_all_segments(model, df)

    # Step 7 — Save everything
    save_artifacts(model, explainer, metrics)

    log.info("=== ai/ml/train.py  DONE ===")
    return model, metrics


if __name__ == "__main__":
    run()