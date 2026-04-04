"""
ingestion/fetch_india_graph.py
==============================
Downloads road graphs for top 50 Indian cities
using graph_from_bbox covering entire district boundaries.

Run:
    # Download all 50 cities
    python -m ingestion.fetch_india_graph --all

    # Single city
    python -m ingestion.fetch_india_graph --city Mumbai

    # Check stats
    python -m ingestion.fetch_india_graph --stats
"""

import os
import json
import logging
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("fetch_india_graph")

DATA_INDIA  = Path("data/india")
CITY_GRAPHS = DATA_INDIA / "city_graphs"
DATA_INDIA.mkdir(parents=True, exist_ok=True)
CITY_GRAPHS.mkdir(parents=True, exist_ok=True)

OSMNX_VER = tuple(int(x) for x in ox.__version__.split(".")[:2])

# ── Top 50 Indian cities — full district bboxes ────────────────────────────────
# bbox format: north, south, east, west (all in decimal degrees)
# Sized to cover entire municipal/district boundary

INDIAN_CITIES = [
    # ── South India ────────────────────────────────────────────────────────────
    {
        "name":  "Bengaluru",
        "state": "Karnataka",
        "bbox":  {"north":13.3500,"south":12.6500,"east":77.9500,"west":77.2500},
        "pop":   "13.6M",
    },
    {
        "name":  "Chennai",
        "state": "Tamil Nadu",
        "bbox":  {"north":13.4000,"south":12.7500,"east":80.4500,"west":79.9500},
        "pop":   "10.9M",
    },
    {
        "name":  "Hyderabad",
        "state": "Telangana",
        "bbox":  {"north":17.7000,"south":17.0500,"east":78.8500,"west":78.1500},
        "pop":   "10.5M",
    },
    {
        "name":  "Kochi",
        "state": "Kerala",
        "bbox":  {"north":10.2500,"south":9.7500, "east":76.5500,"west":76.0500},
        "pop":   "2.1M",
    },
    {
        "name":  "Coimbatore",
        "state": "Tamil Nadu",
        "bbox":  {"north":11.2500,"south":10.7500,"east":77.2500,"west":76.7500},
        "pop":   "2.2M",
    },
    {
        "name":  "Visakhapatnam",
        "state": "Andhra Pradesh",
        "bbox":  {"north":17.9500,"south":17.4500,"east":83.5000,"west":82.9500},
        "pop":   "2.0M",
    },
    {
        "name":  "Madurai",
        "state": "Tamil Nadu",
        "bbox":  {"north":10.1500,"south":9.6500, "east":78.4000,"west":77.9000},
        "pop":   "1.6M",
    },
    {
        "name":  "Mysuru",
        "state": "Karnataka",
        "bbox":  {"north":12.5500,"south":12.0500,"east":76.9500,"west":76.4500},
        "pop":   "1.2M",
    },
    {
        "name":  "Thiruvananthapuram",
        "state": "Kerala",
        "bbox":  {"north":8.7500, "south":8.2500, "east":77.2500,"west":76.7500},
        "pop":   "1.7M",
    },
    {
        "name":  "Tiruchirappalli",
        "state": "Tamil Nadu",
        "bbox":  {"north":11.0500,"south":10.6500,"east":78.9500,"west":78.4500},
        "pop":   "1.1M",
    },

    # ── West India ─────────────────────────────────────────────────────────────
    {
        "name":  "Mumbai",
        "state": "Maharashtra",
        # Full MMR: Thane, Navi Mumbai, Kalyan, Mira Road
        "bbox":  {"north":19.4500,"south":18.7500,"east":73.2500,"west":72.6500},
        "pop":   "20.7M",
    },
    {
        "name":  "Pune",
        "state": "Maharashtra",
        # Covers Pimpri-Chinchwad, Hinjawadi, Kothrud, Hadapsar
        "bbox":  {"north":18.8500,"south":18.2500,"east":74.2000,"west":73.5500},
        "pop":   "6.6M",
    },
    {
        "name":  "Ahmedabad",
        "state": "Gujarat",
        # Covers Gandhinagar, Sanand, Vatva
        "bbox":  {"north":23.3000,"south":22.7500,"east":72.9000,"west":72.3000},
        "pop":   "8.4M",
    },
    {
        "name":  "Surat",
        "state": "Gujarat",
        "bbox":  {"north":21.4500,"south":20.9000,"east":73.1500,"west":72.6000},
        "pop":   "6.6M",
    },
    {
        "name":  "Nagpur",
        "state": "Maharashtra",
        "bbox":  {"north":21.4000,"south":20.8500,"east":79.3500,"west":78.8000},
        "pop":   "2.9M",
    },
    {
        "name":  "Vadodara",
        "state": "Gujarat",
        "bbox":  {"north":22.5500,"south":22.0500,"east":73.4500,"west":72.9500},
        "pop":   "2.3M",
    },
    {
        "name":  "Rajkot",
        "state": "Gujarat",
        "bbox":  {"north":22.5000,"south":22.0500,"east":71.0000,"west":70.5500},
        "pop":   "1.8M",
    },
    {
        "name":  "Nashik",
        "state": "Maharashtra",
        "bbox":  {"north":20.2000,"south":19.7500,"east":74.1000,"west":73.6000},
        "pop":   "1.9M",
    },
    {
        "name":  "Aurangabad",
        "state": "Maharashtra",
        "bbox":  {"north":20.1000,"south":19.6500,"east":75.5500,"west":75.0500},
        "pop":   "1.4M",
    },
    {
        "name":  "Goa",
        "state": "Goa",
        # Covers North + South Goa, Panaji, Margao, Vasco
        "bbox":  {"north":15.8000,"south":15.0000,"east":74.1000,"west":73.6000},
        "pop":   "0.7M",
    },

    # ── North India ────────────────────────────────────────────────────────────
    {
        "name":  "Delhi",
        "state": "Delhi",
        # NCR: Noida, Gurugram, Faridabad, Ghaziabad
        "bbox":  {"north":29.1000,"south":28.2000,"east":77.6000,"west":76.6000},
        "pop":   "32.9M",
    },
    {
        "name":  "Jaipur",
        "state": "Rajasthan",
        "bbox":  {"north":27.2500,"south":26.6000,"east":76.1000,"west":75.5000},
        "pop":   "3.9M",
    },
    {
        "name":  "Lucknow",
        "state": "Uttar Pradesh",
        "bbox":  {"north":27.2000,"south":26.5500,"east":81.3000,"west":80.7000},
        "pop":   "3.8M",
    },
    {
        "name":  "Kanpur",
        "state": "Uttar Pradesh",
        "bbox":  {"north":26.7000,"south":26.2000,"east":80.7000,"west":80.1000},
        "pop":   "3.1M",
    },
    {
        "name":  "Agra",
        "state": "Uttar Pradesh",
        "bbox":  {"north":27.4000,"south":26.9500,"east":78.3500,"west":77.8500},
        "pop":   "1.8M",
    },
    {
        "name":  "Varanasi",
        "state": "Uttar Pradesh",
        "bbox":  {"north":25.5500,"south":25.1500,"east":83.2500,"west":82.7500},
        "pop":   "1.6M",
    },
    {
        "name":  "Meerut",
        "state": "Uttar Pradesh",
        "bbox":  {"north":29.1500,"south":28.7000,"east":77.9500,"west":77.4500},
        "pop":   "1.7M",
    },
    {
        "name":  "Chandigarh",
        "state": "Punjab",
        # Covers Mohali, Panchkula
        "bbox":  {"north":30.9500,"south":30.5500,"east":77.0500,"west":76.6500},
        "pop":   "1.2M",
    },
    {
        "name":  "Amritsar",
        "state": "Punjab",
        "bbox":  {"north":31.8500,"south":31.4500,"east":75.1500,"west":74.6500},
        "pop":   "1.3M",
    },
    {
        "name":  "Ludhiana",
        "state": "Punjab",
        "bbox":  {"north":31.1000,"south":30.6500,"east":76.1000,"west":75.6500},
        "pop":   "1.8M",
    },

    # ── East India ─────────────────────────────────────────────────────────────
    {
        "name":  "Kolkata",
        "state": "West Bengal",
        # Full KMA: Howrah, Salt Lake, New Town, Barrackpore
        "bbox":  {"north":22.9500,"south":22.2000,"east":88.6500,"west":88.0500},
        "pop":   "14.8M",
    },
    {
        "name":  "Bhubaneswar",
        "state": "Odisha",
        # Covers Cuttack
        "bbox":  {"north":20.5500,"south":20.0500,"east":86.1000,"west":85.6000},
        "pop":   "1.0M",
    },
    {
        "name":  "Patna",
        "state": "Bihar",
        "bbox":  {"north":25.8500,"south":25.3500,"east":85.5000,"west":84.9000},
        "pop":   "2.5M",
    },
    {
        "name":  "Ranchi",
        "state": "Jharkhand",
        "bbox":  {"north":23.6000,"south":23.1000,"east":85.6500,"west":85.1000},
        "pop":   "1.4M",
    },
    {
        "name":  "Guwahati",
        "state": "Assam",
        "bbox":  {"north":26.4000,"south":25.9500,"east":92.1000,"west":91.5000},
        "pop":   "1.4M",
    },
    {
        "name":  "Siliguri",
        "state": "West Bengal",
        "bbox":  {"north":26.9000,"south":26.5500,"east":88.6000,"west":88.2500},
        "pop":   "0.7M",
    },

    # ── Central India ──────────────────────────────────────────────────────────
    {
        "name":  "Bhopal",
        "state": "Madhya Pradesh",
        "bbox":  {"north":23.5000,"south":23.0000,"east":77.7000,"west":77.1500},
        "pop":   "2.4M",
    },
    {
        "name":  "Indore",
        "state": "Madhya Pradesh",
        "bbox":  {"north":23.0000,"south":22.4500,"east":76.2000,"west":75.6500},
        "pop":   "3.3M",
    },
    {
        "name":  "Raipur",
        "state": "Chhattisgarh",
        "bbox":  {"north":21.5000,"south":21.0000,"east":82.0000,"west":81.4000},
        "pop":   "1.2M",
    },
    {
        "name":  "Jabalpur",
        "state": "Madhya Pradesh",
        "bbox":  {"north":23.3500,"south":22.9000,"east":80.2500,"west":79.7000},
        "pop":   "1.4M",
    },
    {
        "name":  "Gwalior",
        "state": "Madhya Pradesh",
        "bbox":  {"north":26.4500,"south":25.9500,"east":78.5000,"west":77.9500},
        "pop":   "1.2M",
    },

    # ── Others ─────────────────────────────────────────────────────────────────
    {
        "name":  "Dehradun",
        "state": "Uttarakhand",
        "bbox":  {"north":30.5500,"south":30.1000,"east":78.2500,"west":77.8000},
        "pop":   "0.8M",
    },
    {
        "name":  "Jodhpur",
        "state": "Rajasthan",
        "bbox":  {"north":26.5000,"south":26.0500,"east":73.2500,"west":72.8000},
        "pop":   "1.4M",
    },
    {
        "name":  "Kota",
        "state": "Rajasthan",
        "bbox":  {"north":25.3500,"south":24.9000,"east":76.0500,"west":75.6000},
        "pop":   "1.2M",
    },
    {
        "name":  "Vijayawada",
        "state": "Andhra Pradesh",
        "bbox":  {"north":16.7500,"south":16.2500,"east":80.9000,"west":80.3500},
        "pop":   "1.5M",
    },
    {
        "name":  "Warangal",
        "state": "Telangana",
        "bbox":  {"north":18.2000,"south":17.7500,"east":79.8000,"west":79.3000},
        "pop":   "0.8M",
    },
    {
        "name":  "Tirupati",
        "state": "Andhra Pradesh",
        "bbox":  {"north":13.8500,"south":13.4500,"east":79.6500,"west":79.2500},
        "pop":   "0.5M",
    },
    {
        "name":  "Salem",
        "state": "Tamil Nadu",
        "bbox":  {"north":11.8500,"south":11.4500,"east":78.3500,"west":77.9000},
        "pop":   "0.9M",
    },
    {
        "name":  "Hubli",
        "state": "Karnataka",
        # Covers Dharwad
        "bbox":  {"north":15.5500,"south":15.1000,"east":75.3500,"west":74.9000},
        "pop":   "0.9M",
    },
    {
        "name":  "Mangalore",
        "state": "Karnataka",
        "bbox":  {"north":13.1000,"south":12.7000,"east":75.1000,"west":74.7000},
        "pop":   "0.6M",
    },
]

CITY_BBOXES = {c["name"]: c["bbox"] for c in INDIAN_CITIES}


# ──────────────────────────────────────────────────────────────────────────────
# FETCH SINGLE CITY
# ──────────────────────────────────────────────────────────────────────────────

def fetch_city_graph(city: dict, force: bool = False):
    name = city["name"]
    out  = CITY_GRAPHS / f"{name.lower().replace(' ','_')}.graphml"

    if out.exists() and not force:
        log.info(f"  {name} — already exists")
        try:
            return ox.load_graphml(out)
        except Exception:
            log.warning(f"  {name} — corrupted, re-downloading")

    bbox = city["bbox"]
    log.info(
        f"  Downloading: {name}, {city['state']} "
        f"(pop: {city['pop']}) ..."
    )

    try:
        if OSMNX_VER[0] >= 2:
            G = ox.graph_from_bbox(
                bbox=(
                    bbox["west"],  bbox["south"],
                    bbox["east"],  bbox["north"],
                ),
                network_type="drive",
                retain_all=False,
                simplify=True,
            )
        else:
            G = ox.graph_from_bbox(
                north=bbox["north"], south=bbox["south"],
                east=bbox["east"],   west=bbox["west"],
                network_type="drive",
                retain_all=False,
                simplify=True,
            )

        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)

        G.graph["city"]   = name
        G.graph["state"]  = city["state"]
        G.graph["pop"]    = city["pop"]

        ox.save_graphml(G, out)
        log.info(
            f"  {name}: {len(G.nodes):,} nodes, "
            f"{len(G.edges):,} edges → {out.name}"
        )
        return G

    except Exception as e:
        log.error(f"  {name} FAILED: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# FETCH ALL CITIES
# ──────────────────────────────────────────────────────────────────────────────

def fetch_all_cities(
    cities=None,
    delay: float = 8.0,
    force: bool  = False,
):
    if cities is None:
        cities = INDIAN_CITIES

    results = {}
    success = failed = skipped = 0

    log.info(f"Downloading {len(cities)} Indian city graphs (bbox mode) ...")
    log.info(f"Est. time: {len(cities)*delay/60:.0f} min (with {delay}s delay)")

    for i, city in enumerate(cities):
        name = city["name"]
        out  = CITY_GRAPHS / f"{name.lower().replace(' ','_')}.graphml"

        log.info(f"[{i+1}/{len(cities)}] {name}, {city['state']}")

        if out.exists() and not force:
            skipped += 1
            try:
                G = ox.load_graphml(out)
                results[name] = {
                    "status": "cached",
                    "nodes":  len(G.nodes),
                    "edges":  len(G.edges),
                    "state":  city["state"],
                    "pop":    city["pop"],
                }
            except Exception:
                results[name] = {"status": "cached"}
            continue

        G = fetch_city_graph(city, force=force)

        if G is not None:
            success += 1
            results[name] = {
                "status": "success",
                "nodes":  len(G.nodes),
                "edges":  len(G.edges),
                "state":  city["state"],
                "pop":    city["pop"],
                "bbox":   city["bbox"],
            }
        else:
            failed += 1
            results[name] = {"status": "failed"}

        if i < len(cities) - 1:
            log.info(f"  Waiting {delay}s ...")
            time.sleep(delay)

    # Save metadata
    meta = DATA_INDIA / "city_metadata.json"
    with open(meta, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n{'='*55}")
    log.info(f"DOWNLOAD COMPLETE")
    log.info(f"{'='*55}")
    log.info(f"  Success : {success}")
    log.info(f"  Skipped : {skipped} (already downloaded)")
    log.info(f"  Failed  : {failed}")
    log.info(f"  Metadata: {meta}")
    log.info(f"{'='*55}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CITY INDEX + LOOKUP
# ──────────────────────────────────────────────────────────────────────────────

def get_available_cities() -> list:
    """Returns names of cities with downloaded graphs."""
    return [
        p.stem.replace("_", " ").title()
        for p in sorted(CITY_GRAPHS.glob("*.graphml"))
    ]


def load_city_graph(city_name: str):
    """Loads graph for a city. Returns None if not downloaded."""
    fname = city_name.lower().replace(" ", "_") + ".graphml"
    path  = CITY_GRAPHS / fname
    if not path.exists():
        return None
    log.info(f"Loading: {city_name}")
    return ox.load_graphml(path)


def find_city_for_coordinates(lat: float, lon: float) -> str:
    """
    Returns city name for given GPS coordinates.
    Checks if point falls inside any city bbox.
    Falls back to nearest city center.
    """
    # Check bbox containment first (accurate)
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
        dist = ((lat - clat)**2 + (lon - clon)**2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best      = city["name"]

    return best


# ──────────────────────────────────────────────────────────────────────────────
# STATS
# ──────────────────────────────────────────────────────────────────────────────

def print_stats():
    cities = get_available_cities()
    meta_path = DATA_INDIA / "city_metadata.json"

    if not meta_path.exists() or not cities:
        print("No cities downloaded yet.")
        print("Run: python -m ingestion.fetch_india_graph --all")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    total_nodes = sum(
        v.get("nodes", 0) for v in meta.values()
        if v.get("status") in ("success", "cached")
    )
    total_edges = sum(
        v.get("edges", 0) for v in meta.values()
        if v.get("status") in ("success", "cached")
    )
    states = set(
        v.get("state", "") for v in meta.values()
        if v.get("status") in ("success", "cached")
    )

    print(f"\n{'='*65}")
    print(f"FEAR-FREE NAVIGATOR — ALL-INDIA COVERAGE")
    print(f"{'='*65}")
    print(f"  Cities covered : {len(cities)}")
    print(f"  States covered : {len(states)}")
    print(f"  Total nodes    : {total_nodes:,}")
    print(f"  Total edges    : {total_edges:,}")
    print(f"  Avg edges/city : {total_edges//max(len(cities),1):,}")
    print(f"{'='*65}")
    print(f"\n{'City':<22} {'State':<22} {'Pop':<8} {'Nodes':>8} {'Edges':>8}")
    print(f"{'-'*65}")

    for name, data in sorted(
        meta.items(),
        key=lambda x: x[1].get("edges", 0),
        reverse=True
    ):
        if data.get("status") in ("success", "cached"):
            print(
                f"  {name:<20} "
                f"{data.get('state',''):<22} "
                f"{data.get('pop','?'):<8} "
                f"{data.get('nodes',0):>8,} "
                f"{data.get('edges',0):>8,}"
            )

    print(f"{'='*65}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run(city=None, all=False, stats=False, force=False):
    log.info("=== fetch_india_graph.py START ===")

    if stats:
        print_stats()

    elif city:
        city_data = next(
            (c for c in INDIAN_CITIES
             if c["name"].lower() == city.lower()),
            None
        )
        if not city_data:
            available = [c["name"] for c in INDIAN_CITIES]
            log.error(f"City '{city}' not found.\nAvailable: {available}")
            return
        fetch_city_graph(city_data, force=force)

    elif all:
        fetch_all_cities(force=force)
        print_stats()

    else:
        # Default: top 5 cities
        log.info("No option specified. Downloading top 5 cities ...")
        top5 = [
            c for c in INDIAN_CITIES
            if c["name"] in ["Bengaluru","Mumbai","Delhi","Chennai","Hyderabad"]
        ]
        fetch_all_cities(cities=top5, delay=8.0)
        print_stats()

    log.info("=== fetch_india_graph.py DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download road graphs for Indian cities"
    )
    parser.add_argument("--city",  type=str,        help="Single city name")
    parser.add_argument("--all",   action="store_true", help="Download all 50 cities")
    parser.add_argument("--stats", action="store_true", help="Show coverage stats")
    parser.add_argument("--force", action="store_true", help="Re-download existing")
    args = parser.parse_args()

    run(
        city  = args.city,
        all   = args.all,
        stats = args.stats,
        force = args.force,
    )