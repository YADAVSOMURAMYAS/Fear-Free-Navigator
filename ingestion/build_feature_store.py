"""
ingestion/build_feature_store.py
=================================
Final merge pipeline — combines ALL feature sources into one
master feature table per road segment.

Merges:
    bengaluru_osm_features.csv          ← from fetch_osm.py
    bengaluru_segment_luminosity.csv    ← from fetch_viirs.py
    bengaluru_image_features.csv        ← from fetch_streetview.py
    bengaluru_segment_crime.csv         ← from fetch_crime.py

Output files in data/processed/:
    bengaluru_feature_store.csv         ← master feature table
    bengaluru_feature_store.geojson     ← with geometry for map viz
    feature_summary_stats.json          ← stats for README / evaluation
    bengaluru_scored_graph.graphml      ← graph with safety scores on edges

Run:
    python -m ingestion.build_feature_store

Depends on:
    All 4 previous ingestion files completed successfully.
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, LineString
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("build_feature_store")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_RAW       = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Feature weights for preliminary safety score ──────────────────────────────
# These are used BEFORE the ML model is trained
# ML model (XGBoost) will learn better weights from training data
# But this gives us a working score immediately
FEATURE_WEIGHTS = {
    # Category 1 – Illumination (30%)
    "luminosity_score":      0.18,
    "lamp_count_80m_norm":   0.07,
    "lit_road_bonus":        0.05,

    # Category 2 – Commercial activity (20%)
    "commercial_score":      0.12,
    "emergency_score":       0.08,

    # Category 3 – Footfall / transit (15%)
    "footfall_score":        0.10,
    "transit_score":         0.05,

    # Category 4 – Crime (25%, inverted — high crime = low safety)
    "crime_penalty":         0.25,

    # Category 5 – Physical environment (5%)
    "physical_score":        0.05,

    # Category 6 – Visual / CV features (5%)
    "visual_score":          0.05,
}

# ── Time-of-day multipliers ────────────────────────────────────────────────────
# Applied to commercial/footfall features based on hour
TIME_MULTIPLIERS = {
    "day":      {"commercial": 1.0,  "footfall": 1.0,  "crime": 1.0 },
    "evening":  {"commercial": 0.85, "footfall": 0.85, "crime": 1.15},
    "night":    {"commercial": 0.35, "footfall": 0.40, "crime": 1.35},
    "late":     {"commercial": 0.10, "footfall": 0.15, "crime": 1.50},
}

def get_time_period(hour: int) -> str:
    if   6  <= hour < 20: return "day"
    elif 20 <= hour < 23: return "evening"
    elif 23 <= hour or hour < 3: return "night"
    else:                 return "late"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD ALL FEATURE FILES
# ──────────────────────────────────────────────────────────────────────────────

def load_all_features() -> dict:
    """
    Loads all CSV feature files produced by previous ingestion steps.
    Returns dict of DataFrames.
    Raises clear errors if any required file is missing.
    """
    required = {
        "osm":        DATA_RAW / "bengaluru_osm_features.csv",
        "luminosity": DATA_RAW / "bengaluru_segment_luminosity.csv",
        "crime":      DATA_RAW / "bengaluru_segment_crime.csv",
    }
    optional = {
        "image":      DATA_RAW / "bengaluru_image_features.csv",
    }

    dfs = {}

    # Required files
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required feature file missing: {path}\n"
                f"Run the corresponding ingestion script first."
            )
        dfs[name] = pd.read_csv(path)
        log.info(f"Loaded {name:12s}: {len(dfs[name]):,} rows  ← {path.name}")

    # Optional files — use defaults if missing
    for name, path in optional.items():
        if path.exists():
            dfs[name] = pd.read_csv(path)
            log.info(f"Loaded {name:12s}: {len(dfs[name]):,} rows  ← {path.name}")
        else:
            log.warning(
                f"Optional file missing: {path.name} — "
                f"using default image features."
            )
            dfs[name] = None

    return dfs


# ──────────────────────────────────────────────────────────────────────────────
# 2.  MERGE ALL FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def merge_all_features(dfs: dict) -> pd.DataFrame:
    """
    Merges all feature DataFrames on (u, v, key) edge identifiers.
    Handles missing values with sensible defaults.
    """
    log.info("Merging all feature tables ...")

    # Start with OSM features (has all edges)
    base = dfs["osm"].copy()
    base["u"]   = base["u"].astype(str)
    base["v"]   = base["v"].astype(str)
    base["key"] = base["key"].astype(str)
    log.info(f"Base (OSM): {len(base):,} edges")

    # Merge luminosity
    lum = dfs["luminosity"].copy()
    lum["u"]   = lum["u"].astype(str)
    lum["v"]   = lum["v"].astype(str)
    lum["key"] = lum["key"].astype(str)

    lum_cols = ["u", "v", "key", "mid_lat", "mid_lon",
                "viirs_radiance", "luminosity_norm",
                "luminosity_zone", "luminosity_score"]
    lum = lum[[c for c in lum_cols if c in lum.columns]]

    merged = base.merge(lum, on=["u", "v", "key"], how="left")
    log.info(f"After luminosity merge: {len(merged):,} edges")

    # Merge crime
    crime = dfs["crime"].copy()
    crime["u"]   = crime["u"].astype(str)
    crime["v"]   = crime["v"].astype(str)
    crime["key"] = crime["key"].astype(str)

    crime_cols = ["u", "v", "key", "crime_density",
                  "night_crime_density", "accident_density",
                  "crime_count", "combined_risk_score"]
    crime = crime[[c for c in crime_cols if c in crime.columns]]

    merged = merged.merge(crime, on=["u", "v", "key"], how="left")
    log.info(f"After crime merge: {len(merged):,} edges")

    # Merge image features (optional)
    if dfs["image"] is not None:
        img = dfs["image"].copy()
        img["u"]   = img["u"].astype(str)
        img["v"]   = img["v"].astype(str)
        img["key"] = img["key"].astype(str)

        img_cols = ["u", "v", "key", "brightness_mean", "brightness_min",
                    "darkness_ratio", "greenery_ratio",
                    "contrast_score", "sky_ratio",
                    "has_real_images", "image_source"]
        img = img[[c for c in img_cols if c in img.columns]]

        merged = merged.merge(img, on=["u", "v", "key"], how="left")
        log.info(f"After image merge: {len(merged):,} edges")

    log.info(f"Final merged shape: {merged.shape}")
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 3.  FILL MISSING VALUES
# ──────────────────────────────────────────────────────────────────────────────

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills NaN values with sensible defaults.
    Every feature column must have a value — no NaNs in final store.
    """
    log.info("Filling missing values ...")

    defaults = {
        # Luminosity
        "luminosity_norm":       0.35,
        "luminosity_score":      35.0,
        "luminosity_zone":       "dim",
        "viirs_radiance":        10.0,
        "mid_lat":               12.9716,
        "mid_lon":               77.5946,

        # OSM structural
        "highway_type_enc":      0.42,
        "is_primary_secondary":  0,
        "has_sidewalk":          0,
        "is_dead_end":           0,
        "is_oneway":             0,
        "max_speed_kmh":         30.0,
        "has_road_name":         0,
        "lanes":                 1,
        "length_m":              50.0,
        "travel_time_s":         60.0,
        "u_node_degree":         2,
        "v_node_degree":         2,

        # Proximity counts
        "lamp_count_80m":        0,
        "shop_count_200m":       0,
        "restaurant_count_200m": 0,
        "atm_count_300m":        0,
        "police_count_500m":     0,
        "hospital_count_500m":   0,
        "pharmacy_count_300m":   0,
        "bus_stop_count_300m":   0,
        "cctv_count_150m":       0,
        "construction_nearby":   0,
        "speed_bump_count_100m": 0,

        # Crime
        "crime_density":         0.15,
        "night_crime_density":   0.20,
        "accident_density":      0.10,
        "crime_count":           0,
        "combined_risk_score":   20.0,

        # Image features
        "brightness_mean":       0.35,
        "brightness_min":        0.20,
        "darkness_ratio":        0.30,
        "greenery_ratio":        0.10,
        "contrast_score":        0.50,
        "sky_ratio":             0.20,
        "has_real_images":       False,
        "image_source":          "synthetic",
    }

    for col, default in defaults.items():
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df[col].fillna(default)
            if before > 0:
                log.info(f"  Filled {before:,} NaN in '{col}' with {default}")
        # If column doesn't exist at all, create it
        else:
            df[col] = default

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ENGINEER COMPOSITE FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived composite features from raw columns.
    These are the actual feature vectors fed into XGBoost.
    """
    log.info("Engineering composite features ...")

    # ── Normalise count features to 0–1 ───────────────────────────────────
    df["lamp_count_80m_norm"]    = np.clip(df["lamp_count_80m"]    / 8,  0, 1)
    df["shop_count_200m_norm"]   = np.clip(df["shop_count_200m"]   / 15, 0, 1)
    df["bus_stop_count_300m_norm"]= np.clip(df["bus_stop_count_300m"] / 5, 0, 1)
    df["police_count_500m_norm"] = np.clip(df["police_count_500m"] / 3,  0, 1)
    df["cctv_count_150m_norm"]   = np.clip(df["cctv_count_150m"]   / 5,  0, 1)

    # ── Commercial activity score (Category 2) ─────────────────────────────
    df["commercial_score"] = np.clip(
        0.40 * df["shop_count_200m_norm"] +
        0.30 * np.clip(df["restaurant_count_200m"] / 8, 0, 1) +
        0.20 * np.clip(df["atm_count_300m"]         / 3, 0, 1) +
        0.10 * np.clip(df["pharmacy_count_300m"]    / 3, 0, 1),
        0, 1
    )

    # ── Emergency services score (Category 2) ─────────────────────────────
    df["emergency_score"] = np.clip(
        0.60 * df["police_count_500m_norm"] +
        0.25 * np.clip(df["hospital_count_500m"] / 2, 0, 1) +
        0.15 * df["cctv_count_150m_norm"],
        0, 1
    )

    # ── Footfall score (Category 3) ───────────────────────────────────────
    df["footfall_score"] = np.clip(
        0.45 * df["highway_type_enc"] +
        0.30 * df["is_primary_secondary"] +
        0.25 * df["has_sidewalk"],
        0, 1
    )

    # ── Transit score (Category 3) ────────────────────────────────────────
    df["transit_score"] = df["bus_stop_count_300m_norm"]

    # ── Physical environment score (Category 5) ───────────────────────────
    # Dead ends and construction zones are dangerous
    df["physical_score"] = np.clip(
        0.40 * (1 - df["is_dead_end"]) +
        0.30 * df["has_road_name"] +
        0.20 * (1 - df["construction_nearby"]) +
        0.10 * np.clip(df["lanes"] / 3, 0, 1),
        0, 1
    )

    # ── Lit road bonus (Category 1) ───────────────────────────────────────
    # Extra bonus if OSM explicitly tags road as lit=yes
    df["lit_road_bonus"] = df.get("luminosity_zone", "dim").apply(
        lambda z: 1.0 if z == "bright" else
                  0.6 if z == "moderate" else
                  0.2 if z == "dim" else 0.0
    ) if "luminosity_zone" in df.columns else 0.3

    # ── Visual score (Category 6) ─────────────────────────────────────────
    df["visual_score"] = np.clip(
        0.40 * df["brightness_mean"] +
        0.25 * (1 - df["darkness_ratio"]) +
        0.20 * df["greenery_ratio"] +
        0.15 * df["sky_ratio"],
        0, 1
    )

    # ── Crime penalty (Category 4) ────────────────────────────────────────
    # Inverted — high crime = high penalty = low safety
    df["crime_penalty"] = np.clip(
        0.50 * df["night_crime_density"] +
        0.30 * df["crime_density"] +
        0.20 * df["accident_density"],
        0, 1
    )

    # ── Time encoding features (cyclical) ────────────────────────────────
    # Default to 22:00 (10 PM) — peak unsafe hour
    default_hour = 22
    df["hour_sin"] = np.sin(2 * np.pi * default_hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * default_hour / 24)
    df["is_night"] = 1   # default: night mode

    log.info("Composite features engineered.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5.  COMPUTE PRELIMINARY SAFETY SCORE
# ──────────────────────────────────────────────────────────────────────────────

def compute_preliminary_safety_score(
    df: pd.DataFrame,
    hour: int = 22,
) -> pd.DataFrame:
    """
    Computes a preliminary safety score (0–100) using weighted formula.
    This is used BEFORE ML model training.
    After training, XGBoost predictions replace this score.

    Formula:
        safety = (positive_features - crime_penalty) × time_multiplier × 100

    Positive features: lighting, commercial, footfall, emergency, physical, visual
    Crime penalty: subtracted (inverted feature)
    Time multiplier: adjusts commercial/footfall contribution by time of day
    """
    log.info(f"Computing preliminary safety scores (hour={hour}) ...")

    period = get_time_period(hour)
    mult   = TIME_MULTIPLIERS[period]

    # Update time features for this hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_night"] = int(hour >= 20 or hour <= 5)

    # Positive score components
    positive = (
        FEATURE_WEIGHTS["luminosity_score"]    * (df["luminosity_score"] / 100) +
        FEATURE_WEIGHTS["lamp_count_80m_norm"] * df["lamp_count_80m_norm"] +
        FEATURE_WEIGHTS["lit_road_bonus"]      * df["lit_road_bonus"] +
        FEATURE_WEIGHTS["commercial_score"]    * df["commercial_score"] * mult["commercial"] +
        FEATURE_WEIGHTS["emergency_score"]     * df["emergency_score"] +
        FEATURE_WEIGHTS["footfall_score"]      * df["footfall_score"]  * mult["footfall"] +
        FEATURE_WEIGHTS["transit_score"]       * df["transit_score"] +
        FEATURE_WEIGHTS["physical_score"]      * df["physical_score"] +
        FEATURE_WEIGHTS["visual_score"]        * df["visual_score"]
    )

    # Crime penalty
    penalty = FEATURE_WEIGHTS["crime_penalty"] * df["crime_penalty"] * mult["crime"]

    # Raw score
    raw_score = (positive - penalty + 0.25)  # +0.25 baseline offset

    # Scale to 0–100
    df["safety_score_raw"]  = np.clip(raw_score * 100, 0, 100).round(2)

    # Min-max normalise to use full 0–100 range
    s_min = df["safety_score_raw"].min()
    s_max = df["safety_score_raw"].max()
    if s_max > s_min:
        df["safety_score"] = (
            (df["safety_score_raw"] - s_min) / (s_max - s_min) * 100
        ).round(2)
    else:
        df["safety_score"] = 50.0

    # Safety grade label
    df["safety_grade"] = pd.cut(
        df["safety_score"],
        bins  = [0,  20,  40,  60,  80,  100],
        labels= ["E","D", "C", "B", "A"],
        include_lowest=True,
    )

    log.info(f"\n── Safety Score Summary (hour={hour}) ────────────────")
    log.info(f"  Mean  : {df['safety_score'].mean():.1f}")
    log.info(f"  Median: {df['safety_score'].median():.1f}")
    log.info(f"  Min   : {df['safety_score'].min():.1f}")
    log.info(f"  Max   : {df['safety_score'].max():.1f}")
    log.info(f"  Grade A (80–100): {(df['safety_grade']=='A').sum():,} segments")
    log.info(f"  Grade B (60–80) : {(df['safety_grade']=='B').sum():,} segments")
    log.info(f"  Grade C (40–60) : {(df['safety_grade']=='C').sum():,} segments")
    log.info(f"  Grade D (20–40) : {(df['safety_grade']=='D').sum():,} segments")
    log.info(f"  Grade E (0–20)  : {(df['safety_grade']=='E').sum():,} segments")
    log.info("────────────────────────────────────────────────────")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 6.  INJECT SCORES INTO GRAPH
# ──────────────────────────────────────────────────────────────────────────────

def inject_scores_into_graph(
    G,
    feature_df: pd.DataFrame,
) -> object:
    """
    Writes safety_score back onto every edge in the OSM graph.
    Saves scored graph to data/processed/bengaluru_scored_graph.graphml
    This is the graph used by the routing engine.
    """
    log.info("Injecting safety scores into road graph ...")

    # Build lookup dict for fast access
    score_lookup = {}
    for _, row in feature_df.iterrows():
        score_lookup[(str(row["u"]), str(row["v"]), str(row["key"]))] = {
            "safety_score":  float(row.get("safety_score",  50.0)),
            "safety_grade":  str(row.get("safety_grade",    "C")),
            "crime_density": float(row.get("crime_density", 0.2)),
            "luminosity_score": float(row.get("luminosity_score", 35.0)),
        }

    injected  = 0
    missing   = 0

    for u, v, key, data in G.edges(data=True, keys=True):
        lookup_key = (str(u), str(v), str(key))
        scores = score_lookup.get(lookup_key)

        if scores:
            data.update(scores)
            injected += 1
        else:
            # Default scores for edges not in feature store
            data["safety_score"]     = 40.0
            data["safety_grade"]     = "D"
            data["crime_density"]    = 0.3
            data["luminosity_score"] = 30.0
            missing += 1

    log.info(f"  Injected: {injected:,} edges")
    log.info(f"  Default:  {missing:,} edges (not in feature store)")

    out = DATA_PROCESSED / "bengaluru_scored_graph.graphml"
    ox.save_graphml(G, out)
    log.info(f"Scored graph → {out}")
    return G


# ──────────────────────────────────────────────────────────────────────────────
# 7.  SAVE OUTPUTS
# ──────────────────────────────────────────────────────────────────────────────

def save_feature_store(df: pd.DataFrame) -> None:
    """Saves master feature table in CSV and GeoJSON formats."""

    # ── CSV (used by ML training) ──────────────────────────────────────────
    csv_out = DATA_PROCESSED / "bengaluru_feature_store.csv"
    df.to_csv(csv_out, index=False)
    log.info(f"Feature store CSV → {csv_out}")

    # ── GeoJSON (used by frontend heatmap) ────────────────────────────────
    if "mid_lat" in df.columns and "mid_lon" in df.columns:
        geo_cols = [
            "u", "v", "key",
            "safety_score", "safety_grade",
            "luminosity_score", "commercial_score",
            "crime_penalty", "combined_risk_score",
            "highway_type", "road_name",
            "mid_lat", "mid_lon",
        ]
        geo_df = df[[c for c in geo_cols if c in df.columns]].copy()
        gdf = gpd.GeoDataFrame(
            geo_df,
            geometry=[
                Point(row["mid_lon"], row["mid_lat"])
                for _, row in geo_df.iterrows()
            ],
            crs="EPSG:4326",
        )
        geojson_out = DATA_PROCESSED / "bengaluru_feature_store.geojson"
        gdf.to_file(geojson_out, driver="GeoJSON")
        log.info(f"Feature store GeoJSON → {geojson_out}")


def save_summary_stats(df: pd.DataFrame) -> None:
    """Saves summary statistics for README and evaluation section."""

    stats = {
        "total_segments":      int(len(df)),
        "safety_score": {
            "mean":   round(float(df["safety_score"].mean()),   2),
            "median": round(float(df["safety_score"].median()), 2),
            "std":    round(float(df["safety_score"].std()),    2),
            "min":    round(float(df["safety_score"].min()),    2),
            "max":    round(float(df["safety_score"].max()),    2),
        },
        "grade_distribution": {
            str(g): int((df["safety_grade"] == g).sum())
            for g in ["A", "B", "C", "D", "E"]
        },
        "feature_coverage": {
            "has_luminosity_data": int(df["luminosity_score"].notna().sum()),
            "has_crime_data":      int((df["crime_count"] > 0).sum()),
            "has_real_images":     int(df.get("has_real_images", pd.Series([False])).sum()),
            "has_police_nearby":   int((df["police_count_500m"] > 0).sum()),
            "has_street_lamps":    int((df["lamp_count_80m"] > 0).sum()),
        },
        "top_features_by_mean": {
            col: round(float(df[col].mean()), 4)
            for col in [
                "luminosity_score", "commercial_score",
                "emergency_score",  "footfall_score",
                "crime_penalty",    "visual_score",
            ]
            if col in df.columns
        },
    }

    out = DATA_PROCESSED / "feature_summary_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Summary stats → {out}")
    log.info(json.dumps(stats, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    log.info("=== build_feature_store.py  START ===")

    # Check graph exists
    graph_path = DATA_RAW / "bengaluru_graph.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(
            "Road graph not found.\n"
            "Run: python -m ingestion.fetch_osm"
        )

    # Step 1 – Load all feature files
    dfs = load_all_features()

    # Step 2 – Merge into single table
    merged = merge_all_features(dfs)

    # Step 3 – Fill missing values
    merged = fill_missing_values(merged)

    # Step 4 – Engineer composite features
    merged = engineer_features(merged)

    # Step 5 – Compute preliminary safety score
    merged = compute_preliminary_safety_score(merged, hour=22)

    # Step 6 – Save feature store
    save_feature_store(merged)

    # Step 7 – Save summary stats
    save_summary_stats(merged)

    # Step 8 – Inject scores into graph
    log.info("Loading road graph for score injection ...")
    G = ox.load_graphml(graph_path)
    inject_scores_into_graph(G, merged)

    log.info(f"\n── Final Feature Store ─────────────────────────")
    log.info(f"  Total edges    : {len(merged):,}")
    log.info(f"  Total features : {len(merged.columns)}")
    log.info(f"  Output path    : {DATA_PROCESSED}/bengaluru_feature_store.csv")
    log.info("=== build_feature_store.py  DONE ===")

    return merged


if __name__ == "__main__":
    run()