"""Health check endpoint for the NordSpot API.

GET /health returns the current API status, model version metadata, and
which zones are covered. Used by:
    - Docker Compose healthcheck
    - Load balancer probes (when deployed)
    - Monitoring dashboards to confirm the API is alive
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Response schema for GET /health.

    Attributes:
        status:        "ok" when the API is healthy and able to serve requests.
        model_version: Production model version currently loaded (Stories 5.2+).
                       "not_loaded" until model loading is implemented.
        trained_at:    ISO timestamp of when the Production model was trained.
                       None until model loading is implemented.
        zones_covered: List of bidding zones the API can forecast for.
        api_version:   Semantic API version string.
        timestamp:     UTC timestamp of when this health response was generated.
    """

    status: str
    model_version: str
    trained_at: Optional[str]
    zones_covered: List[str]
    api_version: str
    timestamp: datetime


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    description=(
        "Returns API status, currently loaded model version, and zone coverage. "
        "A 200 response with status='ok' means the API is ready to serve forecasts."
    ),
    tags=["Health"],
)
async def health(request: Request) -> HealthResponse:
    """Return API health status and model metadata."""
    store = request.app.state.model_store
    return HealthResponse(
        status="ok",
        model_version=store.model_version,
        trained_at=store.trained_at,
        zones_covered=["SE1", "SE2", "SE3", "SE4"],
        api_version="v1",
        timestamp=datetime.now(timezone.utc),
    )
