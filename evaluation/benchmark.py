"""
evaluation/benchmark.py
=======================
Comprehensive evaluation suite for the Fear-Free Navigator.
Generates all metrics needed for the submission document.

Run:
    python -m evaluation.benchmark

Output:
    evaluation/results/benchmark_results.csv
    evaluation/results/benchmark_summary.json
    evaluation/results/safety_distribution.png
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

log = logging.getLogger("evaluation.benchmark")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")

RESULTS = Path("evaluation/results")
RESULTS.mkdir(parents=True, exist_ok=True)

# 50 route pairs across Bengaluru
ROUTE_PAIRS = [
    # High danger → safe pairs
    ("Majestic",        (12.9767,77.5713), "Indiranagar",     (12.9718,77.6412)),
    ("KR Market",       (12.9610,77.5762), "Koramangala",     (12.9352,77.6245)),
    ("Shivajinagar",    (12.9839,77.5929), "Jayanagar",       (12.9220,77.5833)),
    ("Chickpete",       (12.9592,77.5673), "HSR Layout",      (12.9116,77.6389)),
    ("Rajajinagar",     (12.9800,77.5700), "MG Road",         (12.9767,77.6009)),
    # Cross-city routes
    ("Hebbal",          (12.9984,77.5931), "Electronic City", (12.8458,77.6603)),
    ("Yeshwanthpur",    (13.0210,77.5500), "Whitefield",      (12.9698,77.7499)),
    ("Malleswaram",     (13.0050,77.5667), "BTM Layout",      (12.9165,77.6101)),
    ("MG Road",         (12.9767,77.6009), "Banashankari",    (12.9255,77.5468)),
    ("Indiranagar",     (12.9718,77.6412), "Majestic",        (12.9767,77.5713)),
    # Residential routes
    ("Jayanagar",       (12.9220,77.5833), "Koramangala",     (12.9352,77.6245)),
    ("HSR Layout",      (12.9116,77.6389), "Indiranagar",     (12.9718,77.6412)),
    ("JP Nagar",        (12.9090,77.5800), "MG Road",         (12.9767,77.6009)),
    ("Banashankari",    (12.9255,77.5468), "Shivajinagar",    (12.9839,77.5929)),
    ("Domlur",          (12.9600,77.6400), "Majestic",        (12.9767,77.5713)),
]

NIGHT_HOURS = [22, 23, 0, 1, 2]
DAY_HOURS   = [9, 12, 15]


def run_single(origin_name, origin, dest_name, dest, alpha=0.7, hour=22):
    from routing.dijkstra import get_dual_routes
    result = get_dual_routes(
        origin[0], origin[1], dest[0], dest[1],
        alpha=alpha, hour=hour,
    )
    if "error" in result:
        return None

    safe = result["safe_route"]
    fast = result["fast_route"]
    comp = result["comparison"]

    return {
        "route":          f"{origin_name} → {dest_name}",
        "hour":           hour,
        "alpha":          alpha,
        "safe_score":     safe["avg_safety_score"],
        "fast_score":     fast["avg_safety_score"],
        "safe_time":      safe["total_time_min"],
        "fast_time":      fast["total_time_min"],
        "safe_dist":      safe["total_dist_km"],
        "fast_dist":      fast["total_dist_km"],
        "safety_gain":    comp["safety_gain_points"],
        "time_cost":      comp["time_penalty_min"],
        "worth_it":       comp["safer_route_worth_it"],
        "safe_dangerous": safe["dangerous_count"],
        "fast_dangerous": fast["dangerous_count"],
        "safe_grade":     safe["safety_grade"],
        "fast_grade":     fast["safety_grade"],
    }


def run_full_benchmark():
    log.info(f"Running benchmark: {len(ROUTE_PAIRS)} routes × hours")

    records = []

    # Test at specific hours that show clear day/night difference
    TEST_HOURS = [
        9,   # Morning
        14,  # Afternoon
        18,  # Evening
        22,  # Night
        0,   # Midnight
    ]

    for origin_name, origin, dest_name, dest in ROUTE_PAIRS:
        for hour in TEST_HOURS:
            log.info(f"  {origin_name} → {dest_name} @ {hour:02d}:00")
            try:
                r = run_single(
                    origin_name, origin,
                    dest_name,   dest,
                    hour=hour
                )
                if r:
                    # Add time period label
                    if   6  <= hour < 17: r["period"] = "day"
                    elif 17 <= hour < 20: r["period"] = "evening"
                    elif 20 <= hour < 24: r["period"] = "night"
                    else:                 r["period"] = "late_night"
                    records.append(r)
            except Exception as e:
                log.warning(f"  Failed: {e}")

    df = pd.DataFrame(records)
    if df.empty:
        log.error("No results.")
        return df

    # ── Stats by period ────────────────────────────────────────────────────
    log.info("\n" + "="*65)
    log.info("BENCHMARK RESULTS BY TIME PERIOD")
    log.info("="*65)

    for period in ["day", "evening", "night", "late_night"]:
        sub = df[df["period"] == period]
        if sub.empty:
            continue
        log.info(
            f"  {period:<12} | "
            f"Safety gain: {sub['safety_gain'].mean():+.2f} pts | "
            f"Time cost: {sub['time_cost'].mean():+.2f} min | "
            f"Safe score: {sub['safe_score'].mean():.1f} | "
            f"Fast score: {sub['fast_score'].mean():.1f} | "
            f"n={len(sub)}"
        )

    log.info("="*65)

    summary = {
        "total_routes_tested": len(df),
        "by_period": {
            period: {
                "avg_safety_gain": round(float(sub["safety_gain"].mean()), 2),
                "avg_time_cost":   round(float(sub["time_cost"].mean()),   2),
                "avg_safe_score":  round(float(sub["safe_score"].mean()),  2),
                "avg_fast_score":  round(float(sub["fast_score"].mean()),  2),
                "pct_worth_it":    round(float(sub["worth_it"].mean()*100),1),
            }
            for period in ["day", "evening", "night", "late_night"]
            for sub in [df[df["period"] == period]]
            if not sub.empty
        }
    }

    df.to_csv(RESULTS / "benchmark_results.csv", index=False)
    with open(RESULTS / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot_period_comparison(df)
    _plot_safety_gain_vs_time(df)
    _plot_score_distribution(df)

    log.info(f"Results → {RESULTS}/")
    return df


def _plot_period_comparison(df):
    """Shows safety gain difference across time periods."""
    periods = ["day", "evening", "night", "late_night"]
    colors  = ["#f59e0b", "#6366f1", "#3b82f6", "#1e293b"]

    gains  = []
    costs  = []
    labels = []

    for p in periods:
        sub = df[df["period"] == p]
        if not sub.empty:
            gains.append(sub["safety_gain"].mean())
            costs.append(sub["time_cost"].mean())
            labels.append(p.replace("_", "\n"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars = axes[0].bar(labels, gains,
                       color=colors[:len(labels)], alpha=0.85)
    axes[0].set_title("Avg Safety Gain by Time Period", fontsize=13)
    axes[0].set_ylabel("Safety gain (points)")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    for bar, val in zip(bars, gains):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05,
                     f"{val:+.2f}", ha="center", fontsize=11)

    bars2 = axes[1].bar(labels, costs,
                        color=colors[:len(labels)], alpha=0.85)
    axes[1].set_title("Avg Time Cost by Time Period", fontsize=13)
    axes[1].set_ylabel("Time cost (minutes)")
    for bar, val in zip(bars2, costs):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f"{val:+.2f}", ha="center", fontsize=11)

    fig.suptitle(
        "Fear-Free Navigator — Time-Aware Routing Performance",
        fontsize=14
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "period_comparison.png", dpi=150)
    plt.close()
    log.info("Plot: period_comparison.png")

def _plot_safety_gain_vs_time(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    night = df[df["hour"].isin(NIGHT_HOURS)]
    day   = df[df["hour"].isin(DAY_HOURS)]

    ax.scatter(night["time_cost"], night["safety_gain"],
               alpha=0.6, color="#6366f1", label="Night (8PM–6AM)", s=60)
    ax.scatter(day["time_cost"],   day["safety_gain"],
               alpha=0.6, color="#f59e0b", label="Day (9AM–3PM)",   s=60)

    ax.axvline(x=5,  color="#ef4444", linestyle="--", alpha=0.5, label="5 min threshold")
    ax.axhline(y=0,  color="#94a3b8", linestyle="-",  alpha=0.3)
    ax.set_xlabel("Time cost (minutes)", fontsize=12)
    ax.set_ylabel("Safety gain (points)", fontsize=12)
    ax.set_title("Safety Gain vs Time Cost — Fear-Free Navigator", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "safety_gain_vs_time.png", dpi=150)
    plt.close()
    log.info("Plot: safety_gain_vs_time.png")


def _plot_score_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df["safe_score"], bins=20, color="#22c55e",
                 alpha=0.7, label="Safe route")
    axes[0].hist(df["fast_score"], bins=20, color="#3b82f6",
                 alpha=0.7, label="Fast route")
    axes[0].set_xlabel("Safety Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Safety Score Distribution")
    axes[0].legend()

    axes[1].hist(df["safety_gain"], bins=20, color="#6366f1", alpha=0.8)
    axes[1].axvline(x=df["safety_gain"].mean(), color="#ef4444",
                    linestyle="--", label=f"Mean: {df['safety_gain'].mean():.1f}")
    axes[1].set_xlabel("Safety Gain (points)")
    axes[1].set_title("Safety Gain Distribution")
    axes[1].legend()

    fig.suptitle("Fear-Free Navigator — Evaluation Results", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS / "score_distribution.png", dpi=150)
    plt.close()
    log.info("Plot: score_distribution.png")


def _plot_night_vs_day(night_df, day_df):
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Avg Safety Gain", "Avg Time Cost", "Safe Route Score", "Fast Route Score"]
    night_vals = [
        night_df["safety_gain"].mean(),
        night_df["time_cost"].mean(),
        night_df["safe_score"].mean(),
        night_df["fast_score"].mean(),
    ]
    day_vals = [
        day_df["safety_gain"].mean(),
        day_df["time_cost"].mean(),
        day_df["safe_score"].mean(),
        day_df["fast_score"].mean(),
    ]

    x   = np.arange(len(categories))
    w   = 0.35
    ax.bar(x - w/2, night_vals, w, label="Night", color="#6366f1", alpha=0.8)
    ax.bar(x + w/2, day_vals,   w, label="Day",   color="#f59e0b", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_title("Night vs Day Performance Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "night_vs_day.png", dpi=150)
    plt.close()
    log.info("Plot: night_vs_day.png")


if __name__ == "__main__":
    run_full_benchmark()