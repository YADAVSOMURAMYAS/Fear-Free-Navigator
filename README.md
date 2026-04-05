# Fear-Free Navigator

Multi-city safety-aware navigation for India. The app computes safe and fast routes using city road graphs, lighting (VIIRS), crime priors, and ML safety scoring.

## Current state

- Multi-city routing is live (city-aware route and heatmap APIs).
- Frontend supports city switching.
- If city is switched during processing, stale requests are cancelled/superseded.
- Large data assets are expected to be downloaded separately (for example from Google Drive).

## Project structure

```text
fear-free-navigator/
├── api/                 FastAPI app and routers
├── routing/             Multi-city routing engine
├── ingestion/           Data generation and refresh scripts
├── ai/ml/               Model features, training, prediction artifacts
├── frontend/            React app (Leaflet map UI)
├── evaluation/          Benchmark scripts
└── data/                Large runtime data (not committed)
	├── india/
	│   ├── city_graphs/ *.graphml
	│   └── features/    *_feature_store.csv (optional but recommended)
	├── raw/
	│   ├── viirs/*.npy
	│   ├── city_crime_index.json
	│   └── city_crime_zones.json
	└── processed/
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- npm

## 1. Clone repository

```bash
git clone https://github.com/YADAVSOMURAMYAS/Fear-Free-Navigator.git
cd fear-free-navigator
```

## 2. Restore data folder from Google Drive

Download the data backup from this Drive link:

https://drive.google.com/file/d/130x6g2zbcqDrNx7v-VSyDnPARNMt5lel/view?usp=sharing

Extract or copy the archive so this repo has at least:

- `data/india/city_graphs/*.graphml` (required)
- `data/india/features/*_feature_store.csv` (recommended for ML scoring)
- `data/raw/viirs/*.npy` (recommended; otherwise luminosity proxy is used)
- `data/raw/city_crime_index.json` and `data/raw/city_crime_zones.json` (recommended)

Minimum required for app to run: `data/india/city_graphs/*.graphml`.

## 3. Run the backend locally

Open PowerShell in the project root and run:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Backend docs: `http://localhost:8000/docs`

## 4. Run the frontend locally

Open a second PowerShell window and run:

```powershell
cd frontend
npm install
npm start
```

Frontend URL: `http://localhost:3000`

## 5. Verify everything is working

From browser or Swagger:

- `GET /health`
- `GET /cities`
- `GET /heatmap/?sample_n=3000&city=Ahmedabad`
- `GET /route/?origin_lat=23.0225&origin_lon=72.5714&dest_lat=23.0473&dest_lon=72.5074&city=Ahmedabad&hour=12&alpha=0.7&mode=car`

Expected:

- `/cities` returns available city names from `data/india/city_graphs`.
- Route response includes `safe_route` and `fast_route`.
- Heatmap returns non-empty `points` for populated city data.

## Notes on fallback behavior

- Missing VIIRS arrays -> highway-type luminosity proxy is used.
- Missing city feature store CSV -> PSI proxy safety score is used.
- Missing crime JSON files -> built-in crime priors/generation are used.

So the system can still run with partial data, but output quality may be lower.

## Optional data refresh workflows

- Rebuild/refresh city graphs:

```powershell
python -m ingestion.fetch_india_graph
```

- Fetch/update VIIRS arrays:

```powershell
python -m ingestion.fetch_viirs_real
```

- Build synthetic multi-city feature stores:

```powershell
python -m ingestion.build_india_features_synthetic
```

- Train India model:

```powershell
python -m ai.ml.train_india
```

## API summary

- `GET /route/` route between two points for a selected city
- `GET /heatmap/` city safety heatmap sample
- `GET /cities/` list supported cities
- `GET /cities/detect` infer city from coordinates
- `GET /score/` single-point score helper
- `POST /report/` submit crowd report

## License

MIT
# 🛡️ Fear-Free Navigator — India (49 Cities)

> **Psychological Safety as a first-class routing metric, scaled to 49 Indian cities.**
> The same AI-powered safe routing engine — now generalized across India with a city-agnostic pipeline and multi-city graph management.

**Google Big Code 2026 Hackathon · Problem Statement #1 · 72-hour submission**

> 📍 **Also see:** [Fear-Free Navigator Bengaluru](https://github.com/YADAVSOMURAMYAS/Fear_Free_Navigator_Bangalore) — the single-city deep-dive with full pipeline details, NASA VIIRS data, and 932K-edge graph.

---

## 🔗 Links

| | |
|---|---|
| 📦 GitHub (this repo) | https://github.com/YADAVSOMURAMYAS/Fear-Free-Navigator |
| 📦 Bengaluru deep-dive repo | https://github.com/YADAVSOMURAMYAS/Fear_Free_Navigator_Bangalore |
| 📁 Data Folder (Google Drive ZIP) | https://drive.google.com/file/d/130x6g2zbcqDrNx7v-VSyDnPARNMt5lel/view?usp=sharing |

> ⚠️ **No live hosted demo** — loading 49 city graphs into RAM requires significant memory and exceeds all free hosting tiers. Run locally using the steps below.

---

## 📸 Screenshots

### Surat — Safe Route (green) vs Fast Route (blue dashed) · 18:00
![Surat Route](screenshots/surat_route.png)

### Kolkata — Safe Route Navigating Howrah · 18:00
![Kolkata Route](screenshots/kolkata_route.png)

### Nashik — Safe Route vs Fast Route · 18:00
![Nashik Route](screenshots/nashik_route.png)

> These screenshots prove the system works on real cities beyond Bengaluru — **Surat, Kolkata, Nashik** all routing correctly with safety-aware paths that visibly differ from the direct fast route.

---

## ❓ What This Repo Does Differently

The **Bengaluru repo** is a single-city deep-dive: one city, full data pipeline, NASA VIIRS satellite, 932K-edge graph, complete ML training.

**This repo** takes the same core engine and scales it to **49 Indian cities**:

| | Bengaluru Repo | This Repo |
|---|---|---|
| Cities | 1 (Bengaluru) | 49 cities across India |
| Data structure | Single graph + feature store | `data/india/city_graphs/*.graphml` per city |
| ML model | XGBoost trained on Bengaluru | XGBoost + LightGBM, trained on India-wide data |
| Workers | Single process | Celery + Redis async workers for parallel city loading |
| Evaluation | Manual log verification | `evaluation/` benchmark scripts |
| Inference | Synchronous | Async with `numba`-accelerated spatial queries |
| Fallback | Requires full data | Graceful degradation — runs with partial data |

---

## 🏙️ Supported Cities (49)

The system supports all major Indian cities and state capitals:

```
Ahmedabad    Bengaluru    Chennai      Delhi        Hyderabad
Kolkata      Mumbai       Pune         Surat        Nashik
Jaipur       Lucknow      Kanpur       Nagpur       Indore
Bhopal       Patna        Ludhiana     Agra         Vadodara
Rajkot       Meerut       Faridabad    Ghaziabad    Kalyan
Vasai-Virar  Aurangabad   Amravati     Vijayawada   Madurai
Coimbatore   Kochi        Visakhapatnam Thiruvananthapuram
Bhubaneswar  Cuttack      Raipur       Guwahati     Jodhpur
Kota         Gwalior      Jabalpur     Tiruchirappalli
Srinagar     Amritsar     Varanasi     Allahabad    Ranchi
Chandigarh
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   FRONTEND (React + Leaflet)                 │
│  City Dropdown · Dark/Light Map · Dual Routes               │
│  Safety Heatmap · Travel Mode · Women Safety Mode           │
│  Time Slider · Turn-by-Turn · Route Comparison Card        │
└───────────────────────────┬─────────────────────────────────┘
							│  HTTP REST (JSON)
┌───────────────────────────▼─────────────────────────────────┐
│                  BACKEND (FastAPI + uvicorn)                  │
│  /route  /heatmap  /cities  /cities/detect  /score  /report │
│  /health   /docs (Swagger UI auto-generated)                 │
└──────┬──────────────────────┬──────────────────┬────────────┘
	   │                      │                  │
┌──────▼──────┐   ┌───────────▼───────┐   ┌──────▼───────────┐
│  Modified   │   │  XGBoost +        │   │  Multi-City      │
│  Dijkstra   │   │  LightGBM ML      │   │  Feature Store   │
│  ×2 paths   │   │  Safety Scorer    │   │  per city CSV    │
└─────────────┘   └───────────────────┘   └──────────────────┘
	   │
┌──────▼──────────────────────────────────────────────────────┐
│              ASYNC WORKERS (Celery + Redis)                  │
│  Background city graph loading · VIIRS fetch updates        │
└─────────────────────────────────────────────────────────────┘
	   │
┌──────▼──────────────────────────────────────────────────────┐
│              DATA LAYER                                       │
│  data/india/city_graphs/*.graphml  (one per city)           │
│  data/india/features/*_feature_store.csv                    │
│  data/raw/viirs/*.npy              (one per city)           │
│  data/raw/city_crime_index.json    (NCRB data)              │
│  data/raw/city_crime_zones.json    (spatial crime zones)    │
└─────────────────────────────────────────────────────────────┘
```

**Key architectural upgrade over the Bengaluru-only version:**

- City graphs are loaded on-demand, not all at startup — the first request for a city triggers its graph load
- Celery workers handle background tasks (VIIRS fetch, graph refresh) without blocking the API
- Graceful degradation: missing VIIRS → highway luminosity proxy; missing feature store → PSI proxy score; missing crime JSON → built-in priors
- `numba`-JIT-compiled spatial queries for faster KD-tree lookups on large city graphs

---

## 🚀 Running Locally

### Prerequisites

```
Python 3.11+
Node.js 18+
8 GB RAM minimum (more cities = more RAM; 49 cities loaded simultaneously requires ~32GB)
```

> **Tip:** You don't need all 49 cities loaded simultaneously. The server loads cities on first request and keeps them in memory. Start with 2–3 cities to test.

### Step 1 — Clone the Repo

```bash
git clone https://github.com/YADAVSOMURAMYAS/Fear-Free-Navigator.git
cd Fear-Free-Navigator
```

### Step 2 — Download and Restore the Data Folder

The `data/` folder contains all city graphs, VIIRS satellite arrays, crime zone JSON files, and ML feature stores. It is not committed to GitHub (too large).

📁 **[Download data ZIP from Google Drive](https://drive.google.com/file/d/130x6g2zbcqDrNx7v-VSyDnPARNMt5lel/view?usp=sharing)**

After downloading, extract the ZIP and place the `data/` folder in the project root:

```
Fear-Free-Navigator/
├── data/
│   ├── india/
│   │   ├── city_graphs/
│   │   │   ├── ahmedabad_graph.graphml
│   │   │   ├── bengaluru_graph.graphml
│   │   │   ├── chennai_graph.graphml
│   │   │   └── ... (one .graphml per city)
│   │   └── features/
│   │       ├── ahmedabad_feature_store.csv
│   │       └── ... (one CSV per city, optional but recommended)
│   ├── raw/
│   │   ├── viirs/
│   │   │   ├── bengaluru_viirs.npy
│   │   │   └── ... (one .npy per city, optional)
│   │   ├── city_crime_index.json
│   │   └── city_crime_zones.json
│   └── processed/
├── api/
├── frontend/
└── ...
```

**Minimum required to run:** `data/india/city_graphs/*.graphml`
The system will fall back to proxy scores for any missing VIIRS or feature store files.

### Step 3 — Set Up Python Environment

```bash
python -m venv venv

venv\Scripts\activate        # Windows
source venv/bin/activate      # Mac/Linux

pip install -r requirements.txt
```

### Step 4 — Set Environment Variables

```bash
cp .env.example .env
# Open .env and fill in:
```

```bash
# .env
GROQ_API_KEY=your_groq_key_here      # Required — Groq LLaMA-3 trip warnings
REDIS_URL=redis://localhost:6379/0   # Required only if using Celery workers
ANTHROPIC_API_KEY=optional           # For advanced LLM features
HOSTED_MODE=false
```

### Step 5 — Run the Backend

Open a terminal in the project root (with venv activated):

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

You will see the API start up:

```
INFO: Started server process
INFO: Uvicorn running on http://0.0.0.0:8000
INFO: Application startup complete.
```

City graphs are loaded **on first request** — the first route request for a city takes ~15–45s while the graph loads. Subsequent requests for the same city are fast (60–100ms).

### Step 6 — Run the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm start
```

### Step 7 — Open in Browser

```
http://localhost:3000
```

Use the **City dropdown** in the top-left to switch between cities. The map will fly to the selected city and load its safety data.

### Step 8 — Verify Everything Works

Test these endpoints in your browser or at `http://localhost:8000/docs`:

```bash
# Health check
GET http://localhost:8000/health

# List all available cities (reads from data/india/city_graphs/)
GET http://localhost:8000/cities

# Heatmap for Ahmedabad
GET http://localhost:8000/heatmap/?sample_n=3000&city=Ahmedabad

# Route in Ahmedabad
GET http://localhost:8000/route/?origin_lat=23.0225&origin_lon=72.5714&dest_lat=23.0473&dest_lon=72.5074&city=Ahmedabad&hour=12&alpha=0.7&mode=car
```

Expected responses:
- `/cities` → list of city names from `data/india/city_graphs/`
- `/route` → JSON with `safe_route` and `fast_route` objects
- `/heatmap` → JSON with non-empty `points` array

---

## 🗺️ Data Structure

### City Graphs (`data/india/city_graphs/`)

One GraphML file per city, downloaded from OpenStreetMap using `osmnx`. Each file contains the full road network for that city's metropolitan area.

```bash
# To rebuild/refresh city graphs from scratch:
python -m ingestion.fetch_india_graph
```

### VIIRS Nighttime Satellite (`data/raw/viirs/`)

One `.npy` file per city — NASA Black Marble VIIRS nighttime light data as a 2D NumPy array. Used to compute the `luminosity_score` feature for every road segment.

```bash
# To fetch/update VIIRS arrays:
python -m ingestion.fetch_viirs_real
```

### Crime Data (`data/raw/`)

```
city_crime_index.json    — NCRB 2022 city-level crime indices for all 49 cities
city_crime_zones.json    — Spatial crime zones with day/night density per city
```

### Feature Stores (`data/india/features/`)

One CSV per city — pre-computed 53-feature store for all road segments. Optional but recommended for full ML accuracy. If missing, the system uses a PSI proxy score computed on-the-fly.

```bash
# To build synthetic feature stores for all cities:
python -m ingestion.build_india_features_synthetic
```

---

## 🧠 ML Model — India-Wide Safety Scorer

**File:** `ai/ml/train_india.py`

```bash
python -m ai.ml.train_india
```

This repo uses both **XGBoost** and **LightGBM** trained on the India-wide multi-city feature dataset.

### Why Both XGBoost and LightGBM?

| Model | Strength | Used for |
|---|---|---|
| XGBoost | Best accuracy on tabular data, SHAP support | Primary safety scorer |
| LightGBM | Faster training, better on high-cardinality features | City-type feature interactions |

The ensemble improves robustness across cities with very different road network densities (dense Mumbai vs sparse Nashik).

### Training Data

```
Cities:         49
Road segments:  ~15–40 million total (varies by city size)
Features:       53 per segment (same as Bengaluru version)
Target:         PSI (Pedestrian Safety Index, 0–100)
```

### Model Artifacts (`ai/ml/artifacts/`)

```
safety_model.pkl         ← Primary XGBoost model
safety_model_lgbm.pkl    ← LightGBM ensemble model
eval_metrics.json        ← R², MAE per city
feature_cols.json        ← 30 selected features
feature_importance.csv   ← SHAP values
```

---

## 🔧 Optional: Data Refresh Workflows

If you want to rebuild the data from scratch instead of using the Google Drive ZIP:

```bash
# 1. Download all 49 city road graphs from OSM (~2–3 hours)
python -m ingestion.fetch_india_graph

# 2. Fetch VIIRS nighttime satellite for all cities (~30 min)
python -m ingestion.fetch_viirs_real

# 3. Build feature stores (synthetic for cities without full POI data)
python -m ingestion.build_india_features_synthetic

# 4. Train the India-wide ML model (~5 min)
python -m ai.ml.train_india
```

---

## 🔌 API Endpoints

All endpoints have interactive docs at `http://localhost:8000/docs`.

```
GET  /route/
	 ?origin_lat=&origin_lon=&dest_lat=&dest_lon=
	 &alpha=0.7&hour=22&mode=car&city=Surat
	 Returns: safe_route + fast_route + comparison + turn-by-turn

GET  /heatmap/
	 ?city=Kolkata&sample_n=3000
	 Returns: 3,000 {lat, lon, score} points for map overlay

GET  /cities/
	 Returns: list of all cities with loaded graphs

GET  /cities/detect
	 ?lat=21.17&lon=72.83
	 Returns: "Surat" (city name from GPS coordinates)

GET  /score/
	 ?lat=22.57&lon=88.36&hour=22
	 Returns: safety_score, grade, crime_density, luminosity

GET  /health
	 Returns: {"status": "ok", "cities_loaded": N}

POST /report/
	 Body: {lat, lon, description, category, severity, city}
	 Returns: confirmation
```

---

## 📁 Project Structure

```
Fear-Free-Navigator/
│
├── ai/
│   └── ml/
│       ├── artifacts/
│       │   ├── safety_model.pkl          ← XGBoost (primary)
│       │   ├── safety_model_lgbm.pkl     ← LightGBM (ensemble)
│       │   ├── eval_metrics.json         ← per-city R², MAE
│       │   ├── feature_cols.json         ← 30 features selected
│       │   └── feature_importance.csv    ← SHAP values
│       ├── train_india.py                ← India-wide training
│       └── predict.py                    ← batch inference
│
├── api/
│   ├── routers/
│   │   ├── route.py      ← /route — dual Dijkstra per city
│   │   ├── heatmap.py    ← /heatmap — sampled city overlay
│   │   ├── cities.py     ← /cities + /cities/detect
│   │   ├── score.py      ← /score — single point
│   │   └── report.py     ← /report — crowdsource reports
│   └── main.py           ← FastAPI app, on-demand city loader
│
├── data/                             ← Downloaded from Google Drive
│   ├── india/
│   │   ├── city_graphs/
│   │   │   ├── ahmedabad_graph.graphml
│   │   │   ├── bengaluru_graph.graphml
│   │   │   └── ... (49 city .graphml files)
│   │   └── features/
│   │       ├── ahmedabad_feature_store.csv
│   │       └── ... (per-city feature CSVs)
│   ├── raw/
│   │   ├── viirs/
│   │   │   └── *.npy               ← NASA satellite per city
│   │   ├── city_crime_index.json   ← NCRB city crime indices
│   │   └── city_crime_zones.json   ← spatial crime zones
│   └── processed/
│
├── evaluation/
│   └── benchmark.py                ← Cross-city routing benchmarks
│
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── components/
│       │   ├── DualRouteMap.jsx    ← Leaflet map + city switching
│       │   ├── SafetyBar.jsx       ← route comparison card
│       │   ├── AlphaSlider.jsx     ← safety preference slider
│       │   └── CitySelector.jsx    ← city dropdown with map fly-to
│       └── hooks/
│           ├── useRoute.js
│           └── useHeatmap.js
│
├── ingestion/
│   ├── fetch_india_graph.py              ← OSM graphs for 49 cities
│   ├── fetch_viirs_real.py               ← NASA VIIRS for all cities
│   ├── build_india_features_synthetic.py ← feature engineering
│   └── city_crime_generator.py           ← crime zone model per city
│
├── routing/
│   ├── city_router.py    ← multi-city graph manager + Dijkstra dispatch
│   └── dijkstra.py       ← modified Dijkstra, composite cost function
│
├── screenshots/
│   ├── surat_route.png
│   ├── kolkata_route.png
│   └── nashik_route.png
│
├── workers/
│   └── tasks.py          ← Celery async tasks (graph refresh, VIIRS update)
│
├── test_cities.py         ← Integration tests for city routing
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 Technical Design

### On-Demand City Loading

Unlike the Bengaluru-only version which loads a single graph at startup, this version loads city graphs on-demand:

1. First request for city X → triggers background graph load (~15–45s)
2. Graph is cached in memory for all subsequent requests
3. The API returns a 202 response on first request and the client retries
4. All 49 cities can be loaded simultaneously if RAM permits (~32GB), or you can limit to active cities

### Graceful Degradation

The system is designed to run with partial data:

| Missing data | Fallback behavior |
|---|---|
| VIIRS `.npy` file | Highway-type luminosity proxy (`hw_lum` from road type encoding) |
| City feature store CSV | On-the-fly PSI proxy from basic OSM attributes |
| Crime JSON files | Built-in NCRB priors by city tier (metro/tier-2/tier-3) |

This means you can test any city that has a `.graphml` file, even without the full feature pipeline.

### Why Celery + Redis?

Downloading and processing city graphs is slow (~15 min per city from scratch). Celery workers handle these jobs in the background without blocking the API. When a new city is requested that isn't pre-loaded, the API triggers a Celery task to load it asynchronously.

For running locally without Redis, the workers are optional — graphs load synchronously on first request instead.

### Numba-Accelerated Spatial Queries

For KD-tree spatial queries (finding POIs within radius of each road segment), the India-wide version uses `numba` JIT compilation — providing 5–10× speedup over pure Python for the feature engineering step across millions of road segments.

---

## 📊 Evaluation

**File:** `evaluation/benchmark.py`

```bash
python evaluation/benchmark.py
```

Runs cross-city routing benchmarks — computes safe vs fast route safety gain across all loaded cities at different hours.

**Integration tests:**

```bash
python test_cities.py
```

Tests route computation for a sample origin/destination pair in each loaded city and verifies the response schema.

---

## 📊 Data Sources

| Source | Data Used | License |
|---|---|---|
| OpenStreetMap | Road networks for all 49 cities, POIs | ODbL (open) |
| NASA GIBS / Black Marble VIIRS | Nighttime satellite brightness per city | Public domain |
| NCRB Crime in India 2022 | City-level crime indices, all cities | Government of India (public) |
| BCP / State Police Annual Reports | Zone-level crime statistics | Public |
| data.gov.in | Road accident hotspots | Government of India (public) |

**All data sources are free and open.**

---
