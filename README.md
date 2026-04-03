# Fear-Free Navigator

A safety-aware routing app for Bengaluru that suggests pedestrian routes based on real-time safety scores — not just travel time. Built for the Google Hackathon.

## What it does

- Scores every road segment in Bengaluru using street-level imagery (CLIP/ViT), crime data, lighting (VIIRS night lights), and OSM features
- Offers a tuneable **alpha** slider: `0 = fastest route`, `1 = safest route`
- Explains *why* a route is safer using SHAP + Claude LLM narration
- Generates heatmaps of unsafe zones across the city
- Asynchronous pipeline via Celery + Redis; results cached in PostGIS

## Architecture

```
fear-free-navigator/
├── api/          # FastAPI backend — routes, scoring, heatmap, report endpoints
├── ai/
│   ├── cv/       # CLIP-based image scoring (brightness, segment features)
│   ├── llm/      # Claude-powered route explanation
│   └── ml/       # LightGBM/XGBoost safety model, SHAP explainer
├── ingestion/    # Data pipeline: OSM, Mapillary, VIIRS, crime data
├── routing/      # Safety-weighted graph routing (osmnx + networkx)
├── workers/      # Celery async task workers
├── data/         # Raw + processed geospatial data (gitignored)
├── db/           # PostGIS schema / migrations
├── evaluation/   # Benchmark against Google Maps routes
└── frontend/     # React + Leaflet map UI
```

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker + Docker Compose
- PostgreSQL with PostGIS extension (provided via Docker)
- Redis (provided via Docker)

### 1. Clone and configure

```bash
git clone https://github.com/your-username/fear-free-navigator.git
cd fear-free-navigator
cp .env.example .env
# Edit .env and fill in your API keys (see Required API Keys below)
```

### 2. Start infrastructure

```bash
docker-compose up -d   # starts PostGIS + Redis
```

### 3. Backend

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### 4. Frontend

```bash
cd frontend
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000)

## Required API keys

| Service | Purpose | Free tier |
|---|---|---|
| [NASA EarthData](https://urs.earthdata.nasa.gov/users/new) | VIIRS night-light rasters | Yes |
| [Mapillary](https://www.mapillary.com/signup) | Street-level imagery | Yes |
| [Google Maps](https://console.cloud.google.com/) | Route benchmarking | $200/mo credit |
| [Anthropic](https://console.anthropic.com/account/keys) | LLM route explanations | Pay-per-use |

Overpass API (OSM) requires no key.

## Data pipeline

Run once to build the feature store:

```bash
python ingestion/build_feature_store.py
```

This fetches OSM features, Mapillary images, VIIRS tiles, and crime data, then trains the safety model.

## Model

- Features: crime density, lighting score, CLIP visual score, lamp/shop/police proximity, CCTV density
- Model: LightGBM + XGBoost ensemble
- Explainability: SHAP values + Claude narration

See `ai/ml/artifacts/eval_metrics.json` for current accuracy metrics.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/route` | Get safety-weighted route |
| `POST` | `/score` | Score a road segment |
| `GET` | `/heatmap` | Safety heatmap GeoJSON |
| `GET` | `/report` | Detailed route safety report |

## License

MIT
