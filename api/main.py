"""NordSpot FastAPI application entry point.

This module creates the FastAPI app, attaches middleware, and registers routers.
It is the single entry point for the ASGI server:

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Routers:
    /health         — liveness probe (no auth)
    /v1/forecast    — 24-hour ahead price forecast (API key auth, Story 5.5)

Middleware:
    CORSMiddleware  — allows cross-origin requests (required for browser clients)
    Request logging — logs method, path, status code, and latency on every call
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.health import router as health_router
from api.loader import load_production_models
from api.model_store import ModelStore
from api.routers.forecast import router as forecast_router

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("nordspot.api")

# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Manage application startup and shutdown."""
    logger.info("NordSpot API starting up  (version=%s)", app.version)
    app.state.model_store = load_production_models()
    yield
    logger.info("NordSpot API shutting down")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="NordSpot",
    description=(
        "Production-grade electricity spot price forecasting API. "
        "Returns 24-hour ahead probabilistic forecasts (point + 90% interval) "
        "for all Swedish bidding zones (SE1-SE4)."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "ATO Energy",
        "url": "https://ato.energy",
    },
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Restrict origins in production; wildcard is fine for development.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Request logging middleware ────────────────────────────────────────────────


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s  status=%d  latency=%.1f ms",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health_router)
app.include_router(forecast_router, prefix="/v1")

# Default state so TestClient (which skips lifespan) has a well-typed store.
# The lifespan overwrites this with a real ModelStore on production startup.
app.state.model_store = ModelStore()
