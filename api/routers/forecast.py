"""POST /v1/forecast — 24-hour ahead probabilistic price forecast.

Story 5.1 stub: validates inputs and returns synthetic data.
Real model loading and prediction are wired up in Stories 5.2–5.4.

Why a stub first?
    Building the full prediction chain (Stories 5.2–5.4) takes several stories.
    The stub lets us verify that routing, schema validation, and error handling
    all work correctly before we connect a real model. Tests written against
    the stub continue to pass unchanged when the real predictor is added.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

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
async def forecast(request: ForecastRequest) -> ForecastResponse:
    """Return a 24-hour probabilistic price forecast.

    Story 5.1 returns synthetic stub data (50 EUR/MWh flat, interval 30–70).
    Stories 5.2–5.4 replace the stub body with real model predictions.
    """
    stub_forecast = [
        HourlyForecast(hour=h, point=50.0, q05=30.0, q95=70.0) for h in range(24)
    ]
    return ForecastResponse(
        zone=request.zone,
        date=request.date,
        model_version="stub-5.1",
        generated_at=datetime.now(timezone.utc),
        forecast=stub_forecast,
    )
