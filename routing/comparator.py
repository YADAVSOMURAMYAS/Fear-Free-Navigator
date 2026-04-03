"""
routing/comparator.py
=====================
Compares safe vs fast routes and generates benchmark results.
Used for evaluation section of the submission document.

Run benchmark:
    python -m routing.comparator
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger("routing.comparator")

EVAL_DIR = Path("evaluation/results")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Known Bengaluru route pairs for benchmarking ───────────────────────────────
BENCHMARK_ROUTES = [
    {
        "name":       "MG Road → Koramangala",
        "origin":     (12.9767, 77.6009),
        "dest":       (12.9352, 77.6245),
        "description":"Busy commercial to residential area",
    },
    {
        "name":       "Majestic → Indiranagar",
        "origin":     (12.9767, 77.5713),
        "dest":       (12.9718, 77.6412),
        "description":"High-crime zone to safer area",
    },
    {
        "name":       "Shivajinagar → Jayanagar",
        "origin":     (12.9839, 77.5929),
        "dest":       (12.9220, 77.5833),
        "description":"North to south Bengaluru",
    },
    {
        "name":       "KR Market → HSR Layout",
        "origin":     (12.9610, 77.5762),
        "dest":       (12.9116, 77.6389),
        "description":"Crime hotspot to safe residential",
    },
    {
        "name":       "Hebbal → MG Road",
        "origin":     (12.9984, 77.5931),
        "dest":       (12.9767, 77.6009),
        "description":"Northern suburb to city center",
    },
    {
        "name":       "Yeshwanthpur → Koramangala",
        "origin":     (13.0210, 77.5500),
        "dest":       (12.9352, 77.6245),
        "description":"Cross-city route",
    },
    {
        "name":       "Malleswaram → BTM Layout",
        "origin":     (13.0050, 77.5667),
        "dest":       (12.9165, 77.6101),
        "description":"Residential to tech hub",
    },
    {
        "name":       "Brigade Road → Banashankari",
        "origin":     (12.9727, 77.6076),
        "dest":       (12.9255, 77.5468),
        "description":"Commercial to residential",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SINGLE ROUTE COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def compare_routes(
    origin_lat: float,
    origin_lon: float,
    dest_lat:   float,
    dest_lon:   float,
    alpha:      float = 0.7,
    hour:       int   = 22,
) -> dict:
    """
    Compares safe vs fast route for a single origin-destination pair.
    Returns comparison metrics.
    """
    from routing.dijkstra import get_dual_routes

    result = get_dual_routes(
        origin_lat, origin_lon,
        dest_lat,   dest_lon,
        alpha=alpha,
        hour=hour,
    )

    if "error" in result:
        return {"error": result["error"]}

    safe = result["safe_route"]
    fast = result["fast_route"]

    return {
        "safe_avg_safety":   safe["avg_safety_score"],
        "fast_avg_safety":   fast["avg_safety_score"],
        "safe_min_safety":   safe["min_safety_score"],
        "fast_min_safety":   fast["min_safety_score"],
        "safe_time_min":     safe["total_time_min"],
        "fast_time_min":     fast["total_time_min"],
        "safe_dist_km":      safe["total_dist_km"],
        "fast_dist_km":      fast["total_dist_km"],
        "safety_gain":       round(safe["avg_safety_score"] - fast["avg_safety_score"], 2),
        "time_cost_min":     round(safe["total_time_min"]   - fast["total_time_min"],   2),
        "safe_dangerous_segs": safe["dangerous_count"],
        "fast_dangerous_segs": fast["dangerous_count"],
        "safe_grade":        safe["safety_grade"],
        "fast_grade":        fast["safety_grade"],
        "worth_it":          result["comparison"]["safer_route_worth_it"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2.  FULL BENCHMARK
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    alpha: float = 0.7,
    hour:  int   = 22,
) -> pd.DataFrame:
    """
    Runs all benchmark route pairs and saves results.
    This is the evaluation data for the submission document.
    """
    log.info(f"Running benchmark: {len(BENCHMARK_ROUTES)} route pairs")
    log.info(f"Settings: alpha={alpha}, hour={hour}")

    results = []

    for i, route in enumerate(BENCHMARK_ROUTES):
        log.info(f"  [{i+1}/{len(BENCHMARK_ROUTES)}] {route['name']}")

        try:
            metrics = compare_routes(
                origin_lat = route["origin"][0],
                origin_lon = route["origin"][1],
                dest_lat   = route["dest"][0],
                dest_lon   = route["dest"][1],
                alpha      = alpha,
                hour       = hour,
            )

            if "error" not in metrics:
                results.append({
                    "route_name":        route["name"],
                    "description":       route["description"],
                    **metrics,
                })
                log.info(
                    f"    Safe: {metrics['safe_avg_safety']:.1f} pts "
                    f"{metrics['safe_time_min']:.1f} min | "
                    f"Fast: {metrics['fast_avg_safety']:.1f} pts "
                    f"{metrics['fast_time_min']:.1f} min | "
                    f"Gain: +{metrics['safety_gain']:.1f} pts"
                )
            else:
                log.warning(f"    Failed: {metrics['error']}")

        except Exception as e:
            log.error(f"    Error: {e}")

    df = pd.DataFrame(results)

    if len(df) == 0:
        log.error("No routes completed successfully.")
        return df

    # ── Summary stats ──────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("BENCHMARK SUMMARY")
    log.info("="*60)
    log.info(f"Routes completed    : {len(df)}/{len(BENCHMARK_ROUTES)}")
    log.info(f"Avg safety gain     : +{df['safety_gain'].mean():.1f} pts")
    log.info(f"Avg time cost       : +{df['time_cost_min'].mean():.1f} min")
    log.info(f"Worth it (<=5 min)  : {df['worth_it'].sum()}/{len(df)} routes")
    log.info(f"Avg dangerous segs  : Safe={df['safe_dangerous_segs'].mean():.1f} "
             f"vs Fast={df['fast_dangerous_segs'].mean():.1f}")
    log.info("="*60)

    # Save results
    out_csv  = EVAL_DIR / "benchmark_results.csv"
    out_json = EVAL_DIR / "benchmark_summary.json"

    df.to_csv(out_csv, index=False)

    summary = {
        "n_routes":           len(df),
        "avg_safety_gain":    round(float(df["safety_gain"].mean()),       2),
        "avg_time_cost_min":  round(float(df["time_cost_min"].mean()),     2),
        "worth_it_count":     int(df["worth_it"].sum()),
        "avg_safe_safety":    round(float(df["safe_avg_safety"].mean()),   2),
        "avg_fast_safety":    round(float(df["fast_avg_safety"].mean()),   2),
        "avg_safe_time_min":  round(float(df["safe_time_min"].mean()),     2),
        "avg_fast_time_min":  round(float(df["fast_time_min"].mean()),     2),
        "alpha":              alpha,
        "hour":               hour,
    }

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Results → {out_csv}")
    log.info(f"Summary → {out_json}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3.  PRINT BENCHMARK TABLE
# ──────────────────────────────────────────────────────────────────────────────

def print_benchmark_table(df: pd.DataFrame):
    """Prints a formatted benchmark results table."""
    if df.empty:
        print("No results to display.")
        return

    print("\n" + "="*90)
    print(f"{'Route':<35} {'Safe':>6} {'Fast':>6} {'Gain':>6} {'TimeCost':>9} {'Grade':>6}")
    print("-"*90)

    for _, row in df.iterrows():
        print(
            f"{row['route_name']:<35} "
            f"{row['safe_avg_safety']:>6.1f} "
            f"{row['fast_avg_safety']:>6.1f} "
            f"{row['safety_gain']:>+6.1f} "
            f"{row['time_cost_min']:>+8.1f}m "
            f"{row['safe_grade']:>6}"
        )

    print("="*90)
    print(
        f"{'AVERAGE':<35} "
        f"{df['safe_avg_safety'].mean():>6.1f} "
        f"{df['fast_avg_safety'].mean():>6.1f} "
        f"{df['safety_gain'].mean():>+6.1f} "
        f"{df['time_cost_min'].mean():>+8.1f}m"
    )
    print("="*90)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )
    log.info("=== routing/comparator.py START ===")

    df = run_benchmark(alpha=0.7, hour=22)
    print_benchmark_table(df)

    log.info("=== routing/comparator.py DONE ===")
    return df


if __name__ == "__main__":
    run()