# evaluation/benchmark_india.py
"""
Runs routing benchmark across all 50 cities.
Tests each city with 3 route pairs at 3 time periods.
"""
from routing.city_router import route_in_city
from ingestion.fetch_india_graph import INDIAN_CITIES, CITY_BBOXES
import pandas as pd
import json
from pathlib import Path

RESULTS = Path("evaluation/results")
RESULTS.mkdir(parents=True, exist_ok=True)

# Test route pairs per city — origin/dest within each city bbox
def get_test_routes(city_name):
    bbox = CITY_BBOXES[city_name]
    clat = (bbox["north"] + bbox["south"]) / 2
    clon = (bbox["east"]  + bbox["west"])  / 2
    span_lat = (bbox["north"] - bbox["south"]) * 0.2
    span_lon = (bbox["east"]  - bbox["west"])  * 0.2
    return [
        # Center to northeast
        (clat - span_lat, clon - span_lon,
         clat + span_lat, clon + span_lon),
        # Center to southwest
        (clat + span_lat, clon + span_lon,
         clat - span_lat, clon - span_lon),
    ]

records = []
TEST_HOURS = [9, 22, 0]

for city in INDIAN_CITIES:
    name  = city["name"]
    routes = get_test_routes(name)

    for olat, olon, dlat, dlon in routes:
        for hour in TEST_HOURS:
            try:
                r = route_in_city(name, olat, olon, dlat, dlon, hour=hour)
                if "error" not in r:
                    records.append({
                        "city":        name,
                        "state":       city["state"],
                        "hour":        hour,
                        "period":      ("day" if 6<=hour<17 else
                                       "night" if 20<=hour<24 else
                                       "late_night"),
                        "safe_score":  r["safe_route"]["avg_safety_score"],
                        "fast_score":  r["fast_route"]["avg_safety_score"],
                        "safety_gain": r["comparison"]["safety_gain_points"],
                        "time_cost":   r["comparison"]["time_penalty_min"],
                        "worth_it":    r["comparison"]["safer_route_worth_it"],
                    })
                    print(f"  ✅ {name} h={hour}: gain=+{r['comparison']['safety_gain_points']:.1f}")
            except Exception as e:
                print(f"  ❌ {name} h={hour}: {e}")

df = pd.DataFrame(records)
df.to_csv(RESULTS / "india_benchmark.csv", index=False)

# Summary
print(f"\n{'='*60}")
print(f"ALL-INDIA BENCHMARK SUMMARY")
print(f"{'='*60}")
print(f"Cities tested    : {df['city'].nunique()}")
print(f"Routes tested    : {len(df)}")
print(f"Avg safety gain  : +{df['safety_gain'].mean():.2f} pts")
print(f"Avg time cost    : +{df['time_cost'].mean():.2f} min")
print(f"Worth it %       : {df['worth_it'].mean()*100:.0f}%")
print(f"\nBy time period:")
for period in ["day","night","late_night"]:
    sub = df[df["period"]==period]
    if len(sub):
        print(f"  {period:<12}: gain=+{sub['safety_gain'].mean():.2f} "
              f"cost=+{sub['time_cost'].mean():.2f}min n={len(sub)}")
print(f"{'='*60}")