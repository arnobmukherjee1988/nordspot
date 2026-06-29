"""POST /v1/forecast — 24-hour ahead probabilistic price forecast.

Story 5.2: model is loaded from MLflow Registry at startup (api.loader)
and stored in app.state.model_store.  The response now includes the real
registry version instead of the hard-coded "stub-5.1" string.

Prediction is still synthetic (50 EUR/MWh flat, interval 30–70).
Real feature retrieval and inference are wired in Stories 5.3–5.4.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from api.schemas import ForecastRequest, ForecastResponse, HourlyForecast

router = APIRouter()


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="24-hour ahead probabilistic price forecast",
    description=(
        "Returns a 24-hour ahead day-ahead electricity price forecast for the "
        "requested Swedish bidding zone and delivery date. "
        "Each hour includes a point forecast (q50) and a 90% prediction interval "
        "[q05, q95]. "
        "**Authentication required** — include your `X-API-Key` header (Story 5.5)."
    ),
    tags=["Forecast"],
)
async def forecast(body: ForecastRequest, request: Request) -> ForecastResponse:
    """Return a 24-hour probabilistic price forecast.

    Story 5.2: model_version is read from app.state.model_store.
    Stub predictions (50 EUR/MWh flat) remain until Stories 5.3–5.4.
    """
    store = request.app.state.model_store
    stub_forecast = [
        HourlyForecast(hour=h, point=50.0, q05=30.0, q95=70.0) for h in range(24)
    ]
    return ForecastResponse(
        zone=body.zone,
        date=body.date,
        model_version=store.model_version,
        generated_at=datetime.now(timezone.utc),
        forecast=stub_forecast,
    )
