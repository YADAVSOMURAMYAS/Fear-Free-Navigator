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

Copy your Drive backup so this repo has at least:

- `data/india/city_graphs/*.graphml` (required)
- `data/india/features/*_feature_store.csv` (recommended for ML scoring)
- `data/raw/viirs/*.npy` (recommended; otherwise luminosity proxy is used)
- `data/raw/city_crime_index.json` and `data/raw/city_crime_zones.json` (recommended)

Minimum required for app to run: `data/india/city_graphs/*.graphml`.

## 3. Backend setup (Windows PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Backend docs: `http://localhost:8000/docs`

## 4. Frontend setup

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
