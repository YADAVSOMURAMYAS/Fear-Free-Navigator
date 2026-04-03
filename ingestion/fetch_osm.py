"""
ingestion/fetch_osm.py
======================
Fetches ALL OSM-derivable safety features for Bengaluru.

Covers:
  Category 1 – Street lamp presence, road lighting infrastructure
  Category 2 – Shops, restaurants, ATMs, hospitals, police, footpaths
  Category 3 – Bus stops, pedestrian sidewalks, road hierarchy
  Category 5 – Road type, dead-ends, CCTVs, construction zones,
                speed bumps, tree canopy tags, isolated paths

Run:
  python -m ingestion.fetch_osm
  
Output files in data/raw/:
  bengaluru_graph.graphml       ← drivable road graph with travel times
  bengaluru_pois.geojson        ← all POIs with category labels
  bengaluru_street_lamps.csv    ← lat/lon of every detected lamp
  bengaluru_graph_features.csv  ← per-edge structural features
"""

import osmnx as ox
import overpy
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from pathlib import Path
import logging
import time
import json

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s"
)
log = logging.getLogger("fetch_osm")

# ── Constants ──────────────────────────────────────────────────────────────────
# Tight bounding box: covers core Bengaluru (excludes far suburbs)
BBOX = {
    "north": 13.0827,
    "south": 12.8340,
    "east":  77.7200,
    "west":  77.4601,
}
# Overpass API string format: south,west,north,east
OVERPASS_BBOX = f"{BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']}"

DATA_RAW = Path("data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)

# ── Road type → encoded score (higher = more footfall, better lit) ─────────────
HIGHWAY_ENCODING = {
    "motorway":      0.95,
    "trunk":         0.90,
    "primary":       0.85,
    "secondary":     0.75,
    "tertiary":      0.60,
    "residential":   0.42,
    "living_street": 0.30,
    "unclassified":  0.22,
    "service":       0.18,
    "track":         0.10,
    "path":          0.08,
    "footway":       0.12,
    "cycleway":      0.14,
}

# OSM sidewalk tags
SIDEWALK_PRESENT = {"both", "left", "right", "yes", "separate"}

# ──────────────────────────────────────────────────────────────────────────────
# 1.  ROAD GRAPH
# ──────────────────────────────────────────────────────────────────────────────

def fetch_road_graph() -> nx.MultiDiGraph:
    out = DATA_RAW / "bengaluru_graph.graphml"
    if out.exists():
        log.info("Graph already exists – loading from disk.")
        return ox.load_graphml(out)

    log.info("Downloading Bengaluru road graph from OSM ...")
    
    # osmnx >= 2.0 uses positional bbox tuple (north, south, east, west)
    # osmnx < 2.0  uses keyword arguments north=, south=, east=, west=
    try:
        # Try new API first (osmnx >= 2.0)
        G = ox.graph_from_bbox(
            bbox=(BBOX["north"], BBOX["south"], BBOX["east"], BBOX["west"]),
            network_type="drive",
            retain_all=False,
            simplify=True,
        )
    except TypeError:
        # Fall back to old API (osmnx < 2.0)
        G = ox.graph_from_bbox(
            north=BBOX["north"], south=BBOX["south"],
            east=BBOX["east"],   west=BBOX["west"],
            network_type="drive",
            retain_all=False,
            simplify=True,
        )

    log.info(f"Raw graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    ox.save_graphml(G, out)
    log.info(f"Graph saved → {out}")
    return G
    """
    Downloads the full drivable road network for Bengaluru.
    Adds edge speeds and travel times.
    Saves to GraphML (preserves all OSM tags).
    """
    out = DATA_RAW / "bengaluru_graph.graphml"
    if out.exists():
        log.info("Graph already exists – loading from disk.")
        return ox.load_graphml(out)

    log.info("Downloading Bengaluru road graph from OSM ...")
    G = ox.graph_from_bbox(
        north=BBOX["north"], south=BBOX["south"],
        east=BBOX["east"],   west=BBOX["west"],
        network_type="drive",
        retain_all=False,
        simplify=True,
        custom_filter=(
            '["highway"~"motorway|trunk|primary|secondary|tertiary'
            '|residential|unclassified|living_street|service"]'
        ),
    )

    log.info(f"Raw graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

    # Add speeds (km/h) and travel times (seconds)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    ox.save_graphml(G, out)
    log.info(f"Graph saved → {out}")
    return G


# ──────────────────────────────────────────────────────────────────────────────
# 2.  POI FETCH  (Categories 1, 2, 3, 5)
# ──────────────────────────────────────────────────────────────────────────────

def _overpass_query(query: str, retries: int = 3) -> overpy.Result:
    """Wraps Overpass calls with retry + backoff."""
    api = overpy.Overpass()
    for attempt in range(retries):
        try:
            return api.query(query)
        except overpy.exception.OverPyException as e:
            wait = 2 ** attempt * 5
            log.warning(f"Overpass error (attempt {attempt+1}): {e}. Retrying in {wait}s …")
            time.sleep(wait)
    raise RuntimeError("Overpass API failed after retries.")


def fetch_all_pois() -> gpd.GeoDataFrame:
    """
    Single Overpass query pulling every safety-relevant POI type.
    
    Category 1 – Illumination:
        highway=street_lamp

    Category 2 – Commercial activity:
        amenity: restaurant, cafe, bar, fast_food, bank, atm, pharmacy,
                 hospital, clinic, police, fire_station
        shop=*  (any shop)
        landuse=retail, commercial

    Category 3 – Footfall / transit:
        highway=bus_stop
        public_transport=stop_position, platform
        amenity=bus_station

    Category 5 – Physical environment:
        man_made=surveillance           (CCTV)
        highway=traffic_calming          (speed bump)
        amenity=parking
        construction=*
        natural=tree_row                 (canopy proxy)
        barrier=*
    """
    out = DATA_RAW / "bengaluru_pois.geojson"
    if out.exists():
        log.info("POIs already exist – loading from disk.")
        return gpd.read_file(out)

    log.info("Querying Bengaluru POIs from Overpass API ...")
    query = f"""
    [out:json][timeout:180];
    (
      /* --- Category 1: Lighting --- */
      node["highway"="street_lamp"]({OVERPASS_BBOX});
      way["highway"="street_lamp"]({OVERPASS_BBOX});

      /* --- Category 2: Commercial --- */
      node["amenity"~"restaurant|cafe|bar|fast_food|bank|atm|pharmacy|hospital|clinic|police|fire_station"]({OVERPASS_BBOX});
      node["shop"]({OVERPASS_BBOX});
      way["landuse"~"retail|commercial"]({OVERPASS_BBOX});

      /* --- Category 3: Transit / footfall --- */
      node["highway"="bus_stop"]({OVERPASS_BBOX});
      node["public_transport"~"stop_position|platform"]({OVERPASS_BBOX});
      node["amenity"="bus_station"]({OVERPASS_BBOX});

      /* --- Category 5: Physical environment --- */
      node["man_made"="surveillance"]({OVERPASS_BBOX});
      node["highway"="speed_bump"]({OVERPASS_BBOX});
      node["traffic_calming"]({OVERPASS_BBOX});
      node["construction"]({OVERPASS_BBOX});
      way["construction"]({OVERPASS_BBOX});
    );
    out center;
    """

    result = _overpass_query(query)
    records = []

    for node in result.nodes:
        tags = node.tags
        records.append({
            "osm_id":    node.id,
            "lat":       float(node.lat),
            "lon":       float(node.lon),
            # Category 1
            "is_street_lamp": tags.get("highway") == "street_lamp",
            # Category 2
            "amenity":    tags.get("amenity", ""),
            "shop":       tags.get("shop", ""),
            "landuse":    tags.get("landuse", ""),
            # Category 3
            "is_bus_stop": (
                tags.get("highway") == "bus_stop"
                or tags.get("public_transport") in ("stop_position", "platform")
                or tags.get("amenity") == "bus_station"
            ),
            # Category 5
            "is_cctv":          tags.get("man_made") == "surveillance",
            "is_speed_bump":    tags.get("highway") == "speed_bump"
                                or bool(tags.get("traffic_calming")),
            "is_construction":  bool(tags.get("construction")),
            "name":       tags.get("name", ""),
        })

    for way in result.ways:
        tags = way.tags
        try:
            lat = float(way.center_lat)
            lon = float(way.center_lon)
        except Exception:
            continue
        records.append({
            "osm_id":    way.id,
            "lat":       lat,
            "lon":       lon,
            "is_street_lamp":   tags.get("highway") == "street_lamp",
            "amenity":          tags.get("amenity", ""),
            "shop":             tags.get("shop", ""),
            "landuse":          tags.get("landuse", "retail"),
            "is_bus_stop":      False,
            "is_cctv":          False,
            "is_speed_bump":    False,
            "is_construction":  bool(tags.get("construction")),
            "name":             tags.get("name", ""),
        })

    gdf = gpd.GeoDataFrame(
        records,
        geometry=[Point(r["lon"], r["lat"]) for r in records],
        crs="EPSG:4326",
    )
    gdf.to_file(out, driver="GeoJSON")
    log.info(f"POIs saved → {out}  ({len(gdf):,} features)")
    return gdf


# ──────────────────────────────────────────────────────────────────────────────
# 3.  STREET LAMP DEDICATED FETCH  (Category 1 – high precision)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_street_lamps() -> pd.DataFrame:
    """
    Dedicated query for street lamps — finer-grained than the bulk POI query.
    Also fetches lit=yes/no tags on road ways for a secondary lighting signal.
    """
    out_lamps = DATA_RAW / "bengaluru_street_lamps.csv"
    out_lit   = DATA_RAW / "bengaluru_lit_roads.csv"

    if out_lamps.exists() and out_lit.exists():
        log.info("Street lamp data already exists – skipping.")
        return pd.read_csv(out_lamps)

    log.info("Fetching street lamps + lit-road tags ...")

    # --- Lamp nodes / ways ---
    lamp_query = f"""
    [out:json][timeout:120];
    (
      node["highway"="street_lamp"]({OVERPASS_BBOX});
      way["highway"="street_lamp"]({OVERPASS_BBOX});
    );
    out center;
    """
    result = _overpass_query(lamp_query)

    lamps = []
    for node in result.nodes:
        lamps.append({
            "lat": float(node.lat),
            "lon": float(node.lon),
            "lamp_type": node.tags.get("lamp_type", "unknown"),
            "support":   node.tags.get("support", "unknown"),
        })
    for way in result.ways:
        try:
            lamps.append({
                "lat": float(way.center_lat),
                "lon": float(way.center_lon),
                "lamp_type": way.tags.get("lamp_type", "unknown"),
                "support":   "way",
            })
        except Exception:
            continue

    lamp_df = pd.DataFrame(lamps)
    lamp_df.to_csv(out_lamps, index=False)
    log.info(f"Street lamps → {out_lamps}  ({len(lamp_df):,})")

    # --- Roads with lit=yes/no tag ---
    lit_query = f"""
    [out:json][timeout:120];
    way["lit"]({OVERPASS_BBOX});
    out center;
    """
    result2 = _overpass_query(lit_query)

    lit_roads = []
    for way in result2.ways:
        try:
            lit_roads.append({
                "osm_id": way.id,
                "lat": float(way.center_lat),
                "lon": float(way.center_lon),
                "lit": way.tags.get("lit", "unknown"),
                "name": way.tags.get("name", ""),
            })
        except Exception:
            continue

    lit_df = pd.DataFrame(lit_roads)
    lit_df.to_csv(out_lit, index=False)
    log.info(f"Lit roads → {out_lit}  ({len(lit_df):,})")
    return lamp_df


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PER-EDGE STRUCTURAL FEATURES  (Categories 3 & 5)
# ──────────────────────────────────────────────────────────────────────────────

def extract_graph_structural_features(G: nx.MultiDiGraph) -> pd.DataFrame:
    """
    Derives structural safety features directly from OSM graph topology.

    Category 3 – Footfall proxies:
        highway_type_enc     road hierarchy score 0–1
        is_primary_secondary binary: major road = more footfall
        has_sidewalk         OSM sidewalk=both/left/right/yes tag
        in_degree / out_degree  (dead-ends have in=1 or out=1)

    Category 5 – Physical environment:
        is_dead_end          True if node has degree=1 (isolated path)
        is_oneway            OSM oneway=yes
        max_speed            posted speed limit (km/h)
        road_name            empty name = unnamed/less-used road
        has_lanes            multi-lane road = more activity
    """
    out = DATA_RAW / "bengaluru_graph_features.csv"
    if out.exists():
        log.info("Graph structural features already exist – skipping.")
        return pd.read_csv(out)

    log.info("Extracting per-edge structural features from graph ...")
    
    # Undirected degree for dead-end detection
    G_undirected = G.to_undirected()
    degree_map = dict(G_undirected.degree())

    records = []
    for u, v, key, data in G.edges(data=True, keys=True):
        hw = data.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0]

        hw_enc = HIGHWAY_ENCODING.get(hw, 0.22)

        # Sidewalk
        sidewalk_tag = data.get("sidewalk", "")
        has_sidewalk = int(str(sidewalk_tag).lower() in SIDEWALK_PRESENT)

        # Dead-end: either endpoint has degree 1
        u_deg = degree_map.get(u, 2)
        v_deg = degree_map.get(v, 2)
        is_dead_end = int(u_deg == 1 or v_deg == 1)

        # Speed limit
        raw_speed = data.get("maxspeed", "")
        try:
            max_speed = float(str(raw_speed).split()[0])
        except (ValueError, AttributeError):
            max_speed = {
                "motorway": 100, "trunk": 80, "primary": 60,
                "secondary": 50, "tertiary": 40, "residential": 30,
            }.get(hw, 30.0)

        # Road name presence
        name = str(data.get("name", "")).strip()
        has_name = int(len(name) > 0)

        # Lanes
        raw_lanes = data.get("lanes", "")
        try:
            lanes = int(str(raw_lanes))
        except (ValueError, TypeError):
            lanes = 1

        records.append({
            "u":                    u,
            "v":                    v,
            "key":                  key,
            "highway_type":         hw,
            "highway_type_enc":     round(hw_enc, 4),
            "is_primary_secondary": int(hw in ("primary", "secondary", "trunk", "motorway")),
            "has_sidewalk":         has_sidewalk,
            "is_dead_end":          is_dead_end,
            "is_oneway":            int(str(data.get("oneway", "no")).lower() == "yes"),
            "max_speed_kmh":        max_speed,
            "has_road_name":        has_name,
            "road_name":            name,
            "lanes":                lanes,
            "length_m":             round(float(data.get("length", 0)), 2),
            "travel_time_s":        round(float(data.get("travel_time", 60)), 2),
            # node degrees (for topology analysis)
            "u_node_degree":        u_deg,
            "v_node_degree":        v_deg,
        })

    df = pd.DataFrame(records)
    df.to_csv(out, index=False)
    log.info(f"Graph structural features → {out}  ({len(df):,} edges)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PROXIMITY FEATURE BUILDER  (Categories 1, 2, 3, 5)
# ──────────────────────────────────────────────────────────────────────────────

def _midpoints_from_graph(G: nx.MultiDiGraph) -> pd.DataFrame:
    """Returns DataFrame of edge midpoints (lat, lon) for spatial joins."""
    rows = []
    for u, v, key, data in G.edges(data=True, keys=True):
        u_lat, u_lon = G.nodes[u]["y"], G.nodes[u]["x"]
        v_lat, v_lon = G.nodes[v]["y"], G.nodes[v]["x"]
        rows.append({
            "u": u, "v": v, "key": key,
            "mid_lat": (u_lat + v_lat) / 2,
            "mid_lon": (u_lon + v_lon) / 2,
        })
    return pd.DataFrame(rows)


def build_proximity_features(
    G: nx.MultiDiGraph,
    pois: gpd.GeoDataFrame,
    lamps: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every road segment midpoint, counts nearby POIs within fixed radii.
    
    Returns DataFrame with columns:
        lamp_count_80m           Category 1
        lit_road_nearby          Category 1
        shop_count_200m          Category 2
        restaurant_count_200m    Category 2
        atm_count_300m           Category 2
        police_count_500m        Category 2
        hospital_count_500m      Category 2
        pharmacy_count_300m      Category 2
        bus_stop_count_300m      Category 3
        cctv_count_150m          Category 5
        construction_nearby      Category 5
        speed_bump_count_100m    Category 5
    """
    out = DATA_RAW / "bengaluru_proximity_features.csv"
    if out.exists():
        log.info("Proximity features already exist – skipping.")
        return pd.read_csv(out)

    log.info("Building proximity features for all edges (this takes ~10 min) ...")

    # Pre-filter POI subsets for speed
    lamp_gdf = gpd.GeoDataFrame(
        lamps,
        geometry=[Point(r.lon, r.lat) for _, r in lamps.iterrows()],
        crs="EPSG:4326",
    )
    shops      = pois[pois["shop"].str.len() > 0]
    restaurants= pois[pois["amenity"].isin(["restaurant","cafe","bar","fast_food"])]
    atms       = pois[pois["amenity"].isin(["atm","bank"])]
    police     = pois[pois["amenity"] == "police"]
    hospitals  = pois[pois["amenity"].isin(["hospital","clinic"])]
    pharmacies = pois[pois["amenity"] == "pharmacy"]
    bus_stops  = pois[pois["is_bus_stop"] == True]
    cctv       = pois[pois["is_cctv"] == True]
    constructs = pois[pois["is_construction"] == True]
    speed_bumps= pois[pois["is_speed_bump"] == True]

    DEG_PER_METRE = 1 / 111_320  # approximate

    def count_within(center_lon, center_lat, gdf, radius_m):
        if gdf.empty:
            return 0
        buf = Point(center_lon, center_lat).buffer(radius_m * DEG_PER_METRE)
        return int(gdf.geometry.within(buf).sum())

    midpoints = _midpoints_from_graph(G)
    records   = []

    for i, row in midpoints.iterrows():
        if i % 5000 == 0:
            log.info(f"  Proximity: {i:,}/{len(midpoints):,}")

        lon, lat = row.mid_lon, row.mid_lat

        records.append({
            "u":   row.u,
            "v":   row.v,
            "key": row.key,
            # Category 1 – Illumination
            "lamp_count_80m":        count_within(lon, lat, lamp_gdf,  80),
            # Category 2 – Commercial
            "shop_count_200m":       count_within(lon, lat, shops,     200),
            "restaurant_count_200m": count_within(lon, lat, restaurants,200),
            "atm_count_300m":        count_within(lon, lat, atms,       300),
            "police_count_500m":     count_within(lon, lat, police,     500),
            "hospital_count_500m":   count_within(lon, lat, hospitals,  500),
            "pharmacy_count_300m":   count_within(lon, lat, pharmacies, 300),
            # Category 3 – Footfall / transit
            "bus_stop_count_300m":   count_within(lon, lat, bus_stops,  300),
            # Category 5 – Physical environment
            "cctv_count_150m":       count_within(lon, lat, cctv,       150),
            "construction_nearby":   int(count_within(lon, lat, constructs, 150) > 0),
            "speed_bump_count_100m": count_within(lon, lat, speed_bumps, 100),
        })

    df = pd.DataFrame(records)
    df.to_csv(out, index=False)
    log.info(f"Proximity features → {out}  ({len(df):,} edges)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

def run():
    log.info("=== fetch_osm.py  START ===")

    # Step 1 – Road graph
    G = fetch_road_graph()

    # Step 2 – All POIs (bulk query)
    pois = fetch_all_pois()

    # Step 3 – Street lamps (dedicated precision query)
    lamps = fetch_street_lamps()

    # Step 4 – Structural features from graph topology
    struct_df = extract_graph_structural_features(G)

    # Step 5 – Proximity counts per edge
    proximity_df = build_proximity_features(G, pois, lamps)

    # Step 6 – Merge structural + proximity into single OSM feature table
    osm_features = struct_df.merge(
        proximity_df, on=["u", "v", "key"], how="left"
    )

    out = DATA_RAW / "bengaluru_osm_features.csv"
    osm_features.to_csv(out, index=False)
    log.info(f"Combined OSM features → {out}  ({len(osm_features):,} edges)")
    log.info("=== fetch_osm.py  DONE ===")

    return G, osm_features


if __name__ == "__main__":
    run()