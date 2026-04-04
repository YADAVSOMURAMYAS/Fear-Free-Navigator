"""
ai/ml/train_india.py
====================
Trains XGBoost safety model on ALL 50 Indian cities combined.
Much more generalized than Bengaluru-only model.

Run:
    python -m ai.ml.train_india
"""

import logging
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
log = logging.getLogger("ml.train_india")

FEAT_DIR  = Path("data/india/features")
ARTIFACTS = Path("ai/ml/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "luminosity_score","luminosity_norm","lamp_count_80m_norm","lit_road_bonus",
    "commercial_score","emergency_score","shop_count_200m_norm","police_count_500m_norm",
    "footfall_score","transit_score","bus_stop_count_300m_norm","is_primary_secondary","has_sidewalk",
    "crime_penalty","crime_density","night_crime_density","accident_density","combined_risk_score",
    "physical_score","is_dead_end","highway_type_enc","has_road_name","lanes","cctv_count_150m_norm","construction_nearby",
    "visual_score","brightness_mean","darkness_ratio","greenery_ratio",
    "hour_sin","hour_cos","is_night",
]

TARGET_COL = "safety_score"


def load_all_cities() -> pd.DataFrame:
    """Loads and combines feature stores from all cities."""
    dfs = []
    csv_files = list(FEAT_DIR.glob("*_feature_store.csv"))
    log.info(f"Found {len(csv_files)} city feature stores")

    for path in sorted(csv_files):
        city = path.stem.replace("_feature_store","").replace("_"," ").title()
        try:
            df = pd.read_csv(path)
            df["city"] = city
            dfs.append(df)
            log.info(f"  {city:<22}: {len(df):,} edges")
        except Exception as e:
            log.warning(f"  {city}: FAILED — {e}")

    if not dfs:
        raise FileNotFoundError(
            "No feature stores found.\n"
            "Run: python -m ingestion.fetch_all_features --all"
        )

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"\nTotal: {len(combined):,} edges from {len(dfs)} cities")
    log.info(f"Cities: {combined['city'].nunique()}")
    return combined


def run():
    log.info("=== ai/ml/train_india.py START ===")

    # Load all cities
    df = load_all_cities()

    # Add missing columns
    for col in FEATURE_COLS:
        if col not in df.columns:
            log.warning(f"Missing: {col} — defaulting to 0")
            df[col] = 0.0

    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL].fillna(50)

    mask = (y > 0.5) & (y < 99.5)
    X, y = X[mask], y[mask]
    log.info(f"Training samples: {len(X):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    log.info("Training XGBoost on all-India data ...")
    model = xgb.XGBRegressor(
        n_estimators          = 1000,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        reg_alpha             = 0.1,
        reg_lambda            = 1.5,
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

    # Evaluate
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2     = r2_score(y_test, y_pred)

    log.info("=" * 55)
    log.info("ALL-INDIA MODEL RESULTS")
    log.info("=" * 55)
    log.info(f"  MAE  : {mae:.3f}")
    log.info(f"  RMSE : {rmse:.3f}")
    log.info(f"  R²   : {r2:.4f}")
    log.info(f"  Cities trained on: {df['city'].nunique()}")
    log.info(f"  Total samples: {len(X_train):,} train, {len(X_test):,} test")
    log.info("=" * 55)

    # SHAP
    log.info("Computing SHAP values ...")
    explainer   = shap.TreeExplainer(model)
    sample      = X_test.iloc[:500]
    shap_values = explainer.shap_values(sample)

    importance = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": np.abs(shap_values).mean(0),
    }).sort_values("importance", ascending=False)

    log.info("\nTop 10 features (All-India model):")
    log.info(importance.head(10).to_string(index=False))

    # Save
    joblib.dump(model,     ARTIFACTS / "india_safety_model.pkl")
    joblib.dump(explainer, ARTIFACTS / "india_shap_explainer.pkl")
    importance.to_csv(ARTIFACTS / "india_feature_importance.csv", index=False)

    metrics = {
        "mae":        round(float(mae),  4),
        "rmse":       round(float(rmse), 4),
        "r2":         round(float(r2),   4),
        "n_cities":   int(df["city"].nunique()),
        "n_train":    int(len(X_train)),
        "n_test":     int(len(X_test)),
        "n_features": len(FEATURE_COLS),
    }
    with open(ARTIFACTS / "india_eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("ALL ARTIFACTS SAVED")
    log.info(f"  Model     : {ARTIFACTS}/india_safety_model.pkl")
    log.info(f"  Metrics   : {ARTIFACTS}/india_eval_metrics.json")
    log.info("=== ai/ml/train_india.py DONE ===")
    return model, metrics


if __name__ == "__main__":
    run()