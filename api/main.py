"""
api/main.py
===========
FastAPI application entry point.

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /              → health check
    GET  /route         → dual route (safe + fast)
    POST /route         → dual route (JSON body)
    GET  /score         → safety score for a location
    POST /report        → crowdsource safety report
    GET  /heatmap       → safety heatmap data
    GET  /docs          → Swagger UI
"""

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routers import route, score, report, heatmap, cities
# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("api.main")

# ── Startup / shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy resources on startup."""
    log.info("Starting Fear-Free Night Navigator API ...")

    # Pre-load graph (slow — do once at startup)
    try:
        from routing.graph import get_graph
        G = get_graph()
        log.info(f"Graph loaded: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
    except Exception as e:
        log.warning(f"Graph not loaded: {e}")

    # Pre-load ML model
    try:
        from ai.ml.predict import load_model
        load_model()
        log.info("ML model loaded.")
    except Exception as e:
        log.warning(f"ML model not loaded: {e}")

    log.info("API ready.")
    yield
    log.info("API shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Fear-Free Night Navigator",
    description = (
        "AI-powered safety routing for Bengaluru. "
        "Optimizes routes using Psychological Safety Score "
        "instead of just travel time."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(route.router)
app.include_router(score.router)
app.include_router(report.router)
app.include_router(heatmap.router)
app.include_router(cities.router)
# ── Static files (frontend) ────────────────────────────────────────────────────
frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount(
        "/static",
        StaticFiles(directory="frontend"),
        name="static",
    )

# ── Root endpoint ──────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
async def root():
    """Health check + API info."""
    try:
        from routing.graph import get_graph
        G            = get_graph()
        graph_loaded = True
        n_edges      = len(G.edges)
    except Exception:
        graph_loaded = False
        n_edges      = 0

    try:
        from ai.ml.predict import load_model
        load_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    return {
        "status":       "ok",
        "app":          "Fear-Free Night Navigator",
        "version":      "1.0.0",
        "city":         "Bengaluru, Karnataka, India",
        "model_loaded": model_loaded,
        "graph_loaded": graph_loaded,
        "n_edges":      n_edges,
        "endpoints": {
            "route":    "/route?origin_lat=12.97&origin_lon=77.60&dest_lat=12.93&dest_lon=77.62",
            "score":    "/score?lat=12.97&lon=77.60",
            "report":   "/report",
            "heatmap":  "/heatmap",
            "docs":     "/docs",
        },
    }


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}