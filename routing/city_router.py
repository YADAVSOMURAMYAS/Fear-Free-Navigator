"""
routing/city_router.py
======================
City-aware routing engine — works for all 50 Indian cities.
Loads correct city graph, injects VIIRS + crime data,
runs time-aware dual Dijkstra.
"""

import json
import logging
import time
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from pathlib import Path

log = logging.getLogger("routing.city_router")

DATA_RAW    = Path("data/raw")
DATA_INDIA  = Path("data/india")
CITY_GRAPHS = DATA_INDIA / "city_graphs"
DATA_PROC   = Path("data/processed")
VIIRS_DIR   = DATA_RAW / "viirs"

_graph_cache      = {}
_graph_cache_time = {}
CACHE_TTL = 3600


def _sf(val, default=0.0):
    try:
        return float(val)
    except:
        return float(default)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD CRIME INDEX
# ──────────────────────────────────────────────────────────────────────────────

def _load_crime_index() -> dict:
    path = DATA_RAW / "city_crime_index.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    from ingestion.fetch_crime_real import CITY_CRIME_INDEX
    return CITY_CRIME_INDEX


def _load_crime_zones(city_name: str) -> list:
    path = DATA_RAW / "city_crime_zones.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get(city_name, [])
    from ingestion.fetch_crime_real import build_crime_zones_for_city
    from ingestion.fetch_india_graph import CITY_BBOXES
    bbox = CITY_BBOXES.get(city_name, {})
    return build_crime_zones_for_city(city_name, bbox)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  LOAD VIIRS
# ──────────────────────────────────────────────────────────────────────────────

def _load_viirs(city_name: str) -> np.ndarray | None:
    fname = city_name.lower().replace(" ", "_") + ".npy"
    path  = VIIRS_DIR / fname
    if path.exists():
        return np.load(path)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 3.  GRAPH LOADING WITH FULL SCORING
# ──────────────────────────────────────────────────────────────────────────────

def load_city_graph(city_name: str):
    """
    Loads and scores city graph.
    Injects VIIRS luminosity + crime density + safety scores.
    Cached in memory for 1 hour.
    """
    now = time.time()
    if city_name in _graph_cache:
        if now - _graph_cache_time.get(city_name, 0) < CACHE_TTL:
            return _graph_cache[city_name]

    # Find graph file
    fname = city_name.lower().replace(" ", "_") + ".graphml"
    path  = CITY_GRAPHS / fname

    if not path.exists():
        log.warning(f"Graph not found: {city_name}. Falling back to Bengaluru.")
        if city_name != "Bengaluru":
            return load_city_graph("Bengaluru")
        # Try processed graph
        for p in [
            DATA_PROC / "bengaluru_scored_graph.graphml",
            Path("data/raw/bengaluru_graph.graphml"),
        ]:
            if p.exists():
                path = p
                break
        else:
            raise FileNotFoundError("No graph found. Run ingestion first.")

    log.info(f"Loading: {city_name}")
    G = ox.load_graphml(path)
    log.info(f"  {city_name}: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

    # Inject all scores
    G = _inject_all_scores(G, city_name)

    _graph_cache[city_name]      = G
    _graph_cache_time[city_name] = now
    return G


def _inject_all_scores(G, city_name: str):
    """
    Injects 3 data sources into graph edges:
    1. VIIRS luminosity (real satellite data)
    2. Crime density (NCRB-based + zone model)
    3. Safety score (ML for Bengaluru, proxy for others)
    """
    from ingestion.fetch_india_graph import CITY_BBOXES

    bbox = CITY_BBOXES.get(city_name)

    # ── 1. VIIRS luminosity ────────────────────────────────────────────────────
    viirs = _load_viirs(city_name)
    if viirs is not None and bbox:
        log.info(f"  Injecting VIIRS luminosity ...")
        h, w      = viirs.shape
        lat_range = bbox["north"] - bbox["south"]
        lon_range = bbox["east"]  - bbox["west"]

        for u, v, data in G.edges(data=True):
            try:
                mid_lat = (_sf(G.nodes[u]["y"]) + _sf(G.nodes[v]["y"])) / 2
                mid_lon = (_sf(G.nodes[u]["x"]) + _sf(G.nodes[v]["x"])) / 2
                row = int((bbox["north"] - mid_lat) / lat_range * h)
                col = int((mid_lon - bbox["west"])  / lon_range * w)
                row = max(0, min(row, h - 1))
                col = max(0, min(col, w - 1))
                data["luminosity_score"] = round(float(viirs[row, col]), 2)
            except:
                data["luminosity_score"] = 35.0
    else:
        log.info(f"  Using highway-type luminosity proxy ...")
        _inject_luminosity_proxy(G)

    # ── 2. Crime density ───────────────────────────────────────────────────────
    log.info(f"  Injecting crime density ...")
    _inject_crime_density(G, city_name, bbox)

    # ── 3. Safety score ────────────────────────────────────────────────────────
    log.info(f"  Computing safety scores ...")
    _inject_safety_scores(G, city_name)

    return G


def _inject_luminosity_proxy(G):
    """Highway-type luminosity when VIIRS unavailable."""
    HW_LUM = {
        "motorway":     88, "trunk":        82, "primary":    75,
        "secondary":    65, "tertiary":     50, "residential":38,
        "living_street":28, "unclassified": 25, "service":    20,
    }
    for u, v, data in G.edges(data=True):
        if "luminosity_score" not in data:
            hw  = data.get("highway", "residential")
            if isinstance(hw, list): hw = hw[0]
            lum = HW_LUM.get(str(hw), 35)
            data["luminosity_score"] = float(
                lum + np.random.uniform(-5, 5)
            )


def _inject_crime_density(G, city_name: str, bbox: dict | None):
    """Assigns crime density to all edges using zone model."""
    zones = _load_crime_zones(city_name)
    base  = _load_crime_index().get(city_name, 0.35)
    DEG   = 1 / 111_320

    for u, v, data in G.edges(data=True):
        if "crime_density" in data:
            try:
                float(data["crime_density"])
                continue
            except:
                pass

        try:
            mid_lat = (_sf(G.nodes[u]["y"]) + _sf(G.nodes[v]["y"])) / 2
            mid_lon = (_sf(G.nodes[u]["x"]) + _sf(G.nodes[v]["x"])) / 2
        except:
            data["crime_density"]       = base
            data["night_crime_density"] = min(0.95, base * 1.35)
            continue

        max_d = base * 0.3
        for zone in zones:
            dist_m = (
                (mid_lat - zone["lat"])**2 +
                (mid_lon - zone["lon"])**2
            ) ** 0.5 / DEG

            if dist_m < zone["r"]:
                impact = zone["d"] * np.exp(
                    -0.5 * (dist_m / (zone["r"] * 0.5))**2
                )
                max_d = max(max_d, float(impact))

        crime = float(np.clip(
            max_d + np.random.normal(0, 0.02),
            0.05, 0.95
        ))
        data["crime_density"]       = round(crime, 3)
        data["night_crime_density"] = round(min(0.95, crime * 1.35), 3)


def _inject_safety_scores(G, city_name: str):
    """
    Uses All-India XGBoost model for all cities.
    Falls back to PSI formula if model unavailable.
    """
    from ai.ml.features import FEATURE_COLS

    # Try India model first
    india_model_path = Path("ai/ml/artifacts/india_safety_model.pkl")
    bengaluru_model  = Path("ai/ml/artifacts/safety_model.pkl")

    model_path = india_model_path if india_model_path.exists() else bengaluru_model

    if model_path.exists():
        import joblib
        model = joblib.load(model_path)
        log.info(f"  Using ML model: {model_path.name}")

        # Check if city has feature store
        feat_path = (
            Path("data/india/features") /
            f"{city_name.lower().replace(' ','_')}_feature_store.csv"
        )

        if feat_path.exists():
            df  = pd.read_csv(feat_path)
            lut = {}
            for _, row in df.iterrows():
                key = (str(int(row["u"])), str(int(row["v"])), str(int(row["key"])))
                # Get available features
                feat_row = {}
                for col in FEATURE_COLS:
                    feat_row[col] = float(row.get(col, 0))
                lut[key] = feat_row

            if lut:
                # Batch predict
                keys     = list(lut.keys())
                feat_df  = pd.DataFrame(list(lut.values()))[FEATURE_COLS].fillna(0)
                scores   = model.predict(feat_df)
                score_map= {k: float(np.clip(s, 0, 100))
                            for k, s in zip(keys, scores)}

                for u, v, k, data in G.edges(data=True, keys=True):
                    key   = (str(u), str(v), str(k))
                    score = score_map.get(key, 40.0)
                    data["safety_score"] = round(score, 2)

                log.info(f"  ML scores injected for {city_name}")
                return

    # Fallback PSI proxy
    log.info(f"  Using PSI proxy for {city_name}")
    _inject_psi_proxy(G)


def _inject_psi_proxy(G):
    """PSI formula fallback when ML model unavailable."""
    HW_ENC = {
        "motorway":0.95,"trunk":0.90,"primary":0.85,
        "secondary":0.75,"tertiary":0.60,"residential":0.42,
        "living_street":0.30,"unclassified":0.22,"service":0.18,
    }
    for u, v, data in G.edges(data=True):
        hw     = data.get("highway","residential")
        if isinstance(hw, list): hw = hw[0]
        hw_enc = HW_ENC.get(str(hw), 0.35)
        lum    = _sf(data.get("luminosity_score", 35), 35) / 100
        crime  = _sf(data.get("crime_density",   0.3), 0.3)
        psi    = float(np.clip(
            28*lum + 22*hw_enc + 18*hw_enc + 15*(1-crime) - 17*crime
            + np.random.normal(0, 2),
            5.0, 95.0
        ))
        data["safety_score"] = round(psi, 2)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  TIME-AWARE EDGE WEIGHTS
# ──────────────────────────────────────────────────────────────────────────────

def apply_edge_weights(
    G,
    alpha:        float = 0.7,
    hour:         int   = 22,
    speed_mult:   float = 1.0,
    blocked_hw:   set   = None,
    preferred_hw: set   = None,
):
    """
    Time-aware composite edge weights with mode support.

    speed_mult: 1.0=car, 0.35=cycling, 0.12=walking
    preferred_hw: these road types get 20% safety bonus
    blocked_hw: these get very high weight (effectively blocked)
    """
    if blocked_hw   is None: blocked_hw   = set()
    if preferred_hw is None: preferred_hw = set()

    if   6  <= hour < 17: period = "day"
    elif 17 <= hour < 20: period = "evening"
    elif 20 <= hour < 24: period = "night"
    else:                 period = "late_night"

    CRIME_EXP   = {"day":1.0,"evening":1.8,"night":3.5,"late_night":6.0}
    LIGHT_FLOOR = {"day":0.0,"evening":5.0,"night":18.0,"late_night":30.0}
    COMM_PEN    = {"day":0.0,"evening":3.0,"night":12.0,"late_night":20.0}

    ce = CRIME_EXP[period]
    lf = LIGHT_FLOOR[period]
    cp = COMM_PEN[period]

    times   = [_sf(d.get("travel_time",60),60) for _,_,d in G.edges(data=True)]
    max_t   = max(times) if times else 300
    min_t   = min(times) if times else 1
    range_t = max(max_t - min_t, 1)

    for u, v, key, data in G.edges(data=True, keys=True):
        hw = data.get("highway","residential")
        if isinstance(hw, list): hw = hw[0]
        hw = str(hw)

        base  = float(np.clip(_sf(data.get("safety_score",  40), 40), 0, 100))
        lum   = _sf(data.get("luminosity_score", 35), 35)
        crime = _sf(data.get("crime_density",   0.15), 0.15)
        comm  = _sf(data.get("commercial_score",  0.3),  0.3)
        lnorm = float(np.clip(lum / 100, 0, 1))

        # Preferred roads get safety bonus
        if hw in preferred_hw:
            base = min(100, base * 1.15)

        # Blocked roads get very high weight
        if hw in blocked_hw:
            data["temporal_safety"]  = 1.0
            data["composite_weight"] = 9999.0
            tt = _sf(data.get("travel_time", 60), 60) * speed_mult
            data["travel_time_mode"] = tt
            continue

        crime_pen = float((crime**0.5) * 15 * ce)
        dark_pen  = float(max(0.0, (1.0 - lnorm) - 0.35) * lf)
        comm_pen  = float(max(0.0, 0.6 - comm) * cp)

        temporal  = float(np.clip(
            base - crime_pen - dark_pen - comm_pen,
            1.0, 100.0
        ))

        # Apply speed multiplier to travel time
        tt        = _sf(data.get("travel_time", 60), 60)
        tt_mode   = tt / max(speed_mult, 0.01)  # walking takes longer
        tnorm     = (tt_mode - min_t) / range_t * 100

        data["temporal_safety"]  = round(temporal, 2)
        data["travel_time_mode"] = round(tt_mode,  2)
        data["composite_weight"] = (
            alpha * (100.0 / max(temporal, 1.0)) +
            (1 - alpha) * tnorm
        )

    return G
# ──────────────────────────────────────────────────────────────────────────────
# 5.  ROUTING
# ──────────────────────────────────────────────────────────────────────────────

def _grade(s):
    return "A" if s>=80 else "B" if s>=60 else "C" if s>=40 else "D" if s>=20 else "E"

def _color(s):
    return "#22c55e" if s>=80 else "#84cc16" if s>=60 else "#f59e0b" if s>=40 else "#ef4444" if s>=20 else "#7f1d1d"


def _route_stats(G, nodes, hour, speed_mult=1.0):
    if not nodes or len(nodes) < 2:
        return {}
    coords, segs, scores, times, lengths = [], [], [], [], []

    for n in nodes:
        coords.append([
            _sf(G.nodes[n].get("y", 0), 0),
            _sf(G.nodes[n].get("x", 0), 0),
        ])

    for u, v in zip(nodes, nodes[1:]):
        d      = G[u][v][0]
        score  = _sf(d.get("temporal_safety", d.get("safety_score", 40)), 40)

        # Use mode-specific travel time if available
        ttime  = _sf(
            d.get("travel_time_mode", d.get("travel_time", 60)),
            60
        )
        length = _sf(d.get("length", 50), 50)

        hw = d.get("highway", "unknown")
        if isinstance(hw, list): hw = hw[0]

        scores.append(score)
        times.append(ttime)
        lengths.append(length)

        segs.append({
            "u":             u,
            "v":             v,
            "safety_score":  round(score,  1),
            "travel_time_s": round(ttime,  1),
            "length_m":      round(length, 1),
            "highway":       str(hw),
            "name":          str(d.get("name", "") or ""),
            "safety_grade":  _grade(score),
            "safety_color":  _color(score),
        })

    avg = float(np.mean(scores))
    return {
        "coords":           coords,
        "segments":         segs,
        "avg_safety_score": round(avg,                   1),
        "min_safety_score": round(float(np.min(scores)), 1),
        "total_time_min":   round(sum(times) / 60,       1),
        "total_dist_km":    round(sum(lengths) / 1000,   2),
        "n_segments":       len(segs),
        "dangerous_count":  sum(1 for s in scores if s < 20),
        "hour":             hour,
        "safety_grade":     _grade(avg),
    }

def route_in_city(
    city_name:  str,
    origin_lat: float,
    origin_lon: float,
    dest_lat:   float,
    dest_lon:   float,
    alpha:      float = 0.7,
    hour:       int   = 22,
    mode:       str   = "car",
) -> dict:
    """
    Routes between two points with mode-specific behavior.

    car        → drives roads, speed 40-80 km/h
    motorcycle → drives roads, slightly faster on small roads
    walking    → avoids highways, speed 5 km/h, safety weight 0.9
    cycling    → avoids motorways, speed 15 km/h, safety weight 0.85
    """
    log.info(
        f"Routing: {city_name} | "
        f"{origin_lat:.4f},{origin_lon:.4f} → "
        f"{dest_lat:.4f},{dest_lon:.4f} | "
        f"α={alpha} h={hour} mode={mode}"
    )

    G = load_city_graph(city_name)

    # ── Mode-specific settings ─────────────────────────────────────────────────
    MODE_CONFIG = {
        "car": {
            "alpha":         alpha,
            "speed_mult":    1.0,
            "blocked_hw":    [],
            "preferred_hw":  ["primary","secondary","trunk","motorway"],
        },
        "motorcycle": {
            "alpha":         max(alpha, 0.75),
            "speed_mult":    0.9,
            "blocked_hw":    [],
            "preferred_hw":  ["primary","secondary","tertiary"],
        },
        "walking": {
            "alpha":         0.90,   # walkers care most about safety
            "speed_mult":    0.12,   # 5 km/h vs 40 km/h car = 0.12x
            "blocked_hw":    ["motorway","trunk","motorway_link","trunk_link"],
            "preferred_hw":  ["footway","path","pedestrian","residential","living_street"],
        },
        "cycling": {
            "alpha":         0.85,
            "speed_mult":    0.35,   # 15 km/h vs 40 km/h = 0.35x
            "blocked_hw":    ["motorway","trunk","motorway_link"],
            "preferred_hw":  ["cycleway","residential","tertiary","living_street"],
        },
    }

    cfg         = MODE_CONFIG.get(mode, MODE_CONFIG["car"])
    mode_alpha  = cfg["alpha"]
    speed_mult  = cfg["speed_mult"]
    blocked_hw  = set(cfg["blocked_hw"])
    preferred_hw= set(cfg["preferred_hw"])

    # ── Apply mode-specific edge weights ───────────────────────────────────────
    G = apply_edge_weights(
        G,
        alpha      = mode_alpha,
        hour       = hour,
        speed_mult = speed_mult,
        blocked_hw = blocked_hw,
        preferred_hw=preferred_hw,
    )

    try:
        orig = ox.nearest_nodes(G, origin_lon, origin_lat)
        dest = ox.nearest_nodes(G, dest_lon,   dest_lat)
    except Exception as e:
        return {"error": f"Could not snap to road: {e}"}

    # For walking/cycling, filter out blocked highways before routing
    if blocked_hw:
        G_filtered = _filter_graph_for_mode(G, blocked_hw, preferred_hw)
    else:
        G_filtered = G

    try:
        safe_nodes = nx.shortest_path(
            G_filtered, orig, dest, weight="composite_weight"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Fallback to full graph if filtered has no path
        try:
            safe_nodes = nx.shortest_path(
                G, orig, dest, weight="composite_weight"
            )
        except Exception:
            return {"error": f"No safe route found in {city_name}"}

    try:
        fast_nodes = nx.shortest_path(
            G_filtered, orig, dest, weight="travel_time"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        try:
            fast_nodes = nx.shortest_path(
                G, orig, dest, weight="travel_time"
            )
        except Exception:
            return {"error": f"No fast route found in {city_name}"}

    safe = _route_stats(G, safe_nodes, hour, speed_mult)
    fast = _route_stats(G, fast_nodes, hour, speed_mult)

    if not safe or not fast:
        return {"error": "Route computation returned empty result."}

    time_pen    = safe["total_time_min"] - fast["total_time_min"]
    safety_gain = safe["avg_safety_score"] - fast["avg_safety_score"]

    log.info(
        f"  [{mode}] Safe: {safe['avg_safety_score']:.1f}pts "
        f"{safe['total_time_min']:.1f}min | "
        f"Fast: {fast['avg_safety_score']:.1f}pts "
        f"{fast['total_time_min']:.1f}min | "
        f"Gain: +{safety_gain:.1f} Cost: +{time_pen:.1f}min"
    )

    return {
        "safe_route":  safe,
        "fast_route":  fast,
        "comparison": {
            "time_penalty_min":    round(time_pen,     1),
            "safety_gain_points":  round(safety_gain,  1),
            "recommendation":     (
                "Take the safer route — minimal time cost."
                if time_pen <= 5 else
                "Safer route adds significant time. Your choice."
            ),
            "safer_route_worth_it": time_pen <= 5 or safety_gain >= 15,
        },
        "city":        city_name,
        "mode":        mode,
        "alpha":       mode_alpha,
        "hour":        hour,
        "origin":      {"lat": origin_lat, "lon": origin_lon},
        "destination": {"lat": dest_lat,   "lon": dest_lon},
    }


def _filter_graph_for_mode(
    G,
    blocked_hw:   set,
    preferred_hw: set,
) -> nx.MultiDiGraph:
    """
    Returns subgraph excluding blocked highway types.
    Used for walking/cycling to avoid motorways.
    """
    edges_to_keep = []
    for u, v, k, data in G.edges(data=True, keys=True):
        hw = data.get("highway", "residential")
        if isinstance(hw, list): hw = hw[0]
        hw = str(hw)
        if hw not in blocked_hw:
            edges_to_keep.append((u, v, k))

    H = G.edge_subgraph(edges_to_keep).copy()
    return H


def get_available_cities() -> list:
    return sorted([
        p.stem.replace("_", " ").title()
        for p in CITY_GRAPHS.glob("*.graphml")
    ])


def detect_city(lat: float, lon: float) -> str:
    """Auto-detects city from GPS coordinates."""
    try:
        from ingestion.fetch_india_graph import INDIAN_CITIES
        
        # Check bbox containment first
        for city in INDIAN_CITIES:
            bbox = city["bbox"]
            if (bbox["south"] <= lat <= bbox["north"] and
                    bbox["west"]  <= lon <= bbox["east"]):
                return city["name"]

        # Fallback: nearest city center
        best      = "Bengaluru"
        best_dist = float("inf")
        for city in INDIAN_CITIES:
            bbox = city["bbox"]
            clat = (bbox["north"] + bbox["south"]) / 2
            clon = (bbox["east"]  + bbox["west"])  / 2
            dist = ((lat - clat)**2 + (lon - clon)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best      = city["name"]

        return best

    except Exception as e:
        log.error(f"detect_city error: {e}")
        return "Bengaluru"