from routing.city_router import route_in_city

cities = [
    ("Mumbai",     19.0760, 72.8777, 19.0596, 72.8295),
    ("Delhi",      28.6139, 77.2090, 28.5355, 77.3910),
    ("Chennai",    13.0827, 80.2707, 13.0500, 80.2100),
    ("Hyderabad",  17.3850, 78.4867, 17.4400, 78.3700),
    ("Kolkata",    22.5726, 88.3639, 22.6200, 88.4200),
    ("Pune",       18.5204, 73.8567, 18.4600, 73.9200),
    ("Ahmedabad",  23.0225, 72.5714, 23.0700, 72.5200),
    ("Jaipur",     26.9124, 75.7873, 26.8500, 75.8500),
    ("Lucknow",    26.8467, 80.9462, 26.8000, 81.0000),
    ("Chandigarh", 30.7333, 76.7794, 30.6800, 76.8500),
]

print("=" * 70)
print(f"{'City':<15} {'Safe':>6} {'Fast':>6} {'Gain':>6} {'Time':>8} {'Grade':>6}")
print("-" * 70)

for city, olat, olon, dlat, dlon in cities:
    try:
        r = route_in_city(city, olat, olon, dlat, dlon, hour=22)
        if "error" not in r:
            safe  = r["safe_route"]["avg_safety_score"]
            fast  = r["fast_route"]["avg_safety_score"]
            gain  = r["comparison"]["safety_gain_points"]
            cost  = r["comparison"]["time_penalty_min"]
            grade = r["safe_route"]["safety_grade"]
            print(f"{city:<15} {safe:>6.1f} {fast:>6.1f} {gain:>+6.1f} {cost:>+7.1f}m {grade:>6}")
        else:
            print(f"{city:<15} ERROR: {r['error']}")
    except Exception as e:
        print(f"{city:<15} EXCEPTION: {e}")

print("=" * 70)