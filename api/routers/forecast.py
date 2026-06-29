"""POST /v1/forecast - 24-hour ahead probabilistic price forecast.

Story 5.5: API key authentication added via Depends(verify_api_key).

Request flow:
    0. verify_api_key()             -> HTTP 401 if X-API-Key missing / invalid
    1. Check store.is_ready         -> HTTP 503 if no Production model loaded
    2. get_inference_features()     -> 24-row feature DataFrame (Story 5.3)
    3. run_inference()              -> ens_q05, ens_q50, ens_q95 (Story 5.4)
    4. Assemble ForecastResponse with 24 HourlyForecast objects
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import verify_api_key
from api.features import get_inference_features
from api.predictor import run_inference
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
        "**Authentication required** - include your `X-API-Key` header."
    ),
    tags=["Forecast"],
    dependencies=[Depends(verify_api_key)],
)
async def forecast(body: ForecastRequest, request: Request) -> ForecastResponse:
    """Return a 24-hour probabilistic price forecast.

    Returns HTTP 401 when X-API-Key is missing or invalid.
    Returns HTTP 503 when no Production model has been loaded (i.e. training
    has not yet completed or ``register_and_promote()`` was not called).
    """
    store = request.app.state.model_store

    if not store.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "No Production model is loaded. "
                "Complete a training run and call register_and_promote(), "
                "then restart the API to load the new Production version."
            ),
        )

    # -- Feature retrieval -------------------------------------------------
    features_df = get_inference_features(body.zone, body.date)

    # -- Inference ---------------------------------------------------------
    preds = run_inference(features_df)

    # -- Assemble response -------------------------------------------------
    hourly = [
        HourlyForecast(
            hour=h,
            point=round(float(preds.iloc[h]["ens_q50"]), 2),
            q05=round(float(preds.iloc[h]["ens_q05"]), 2),
            q95=round(float(preds.iloc[h]["ens_q95"]), 2),
        )
        for h in range(24)
    ]

    return ForecastResponse(
        zone=body.zone,
        date=body.date,
        model_version=store.model_version,
        generated_at=datetime.now(timezone.utc),
        forecast=hourly,
    )
