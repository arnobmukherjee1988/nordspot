"""POST /v1/forecast - 24-hour ahead probabilistic price forecast.

Story 5.5: API key authentication added via Depends(verify_api_key).
Story 6.1: Predictions persisted to TimeDB after each successful inference
           so monitoring/drift_detector.py can compare them against actuals.

Request flow:
    0. verify_api_key()             -> HTTP 401 if X-API-Key missing / invalid
    1. Check store.is_ready         -> HTTP 503 if no Production model loaded
    2. get_inference_features()     -> 24-row feature DataFrame (Story 5.3)
    3. run_inference()              -> ens_q05, ens_q50, ens_q95 (Story 5.4)
    4. Persist predictions to TimeDB (non-blocking on failure)
    5. Assemble ForecastResponse with 24 HourlyForecast objects
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import verify_api_key
from api.features import get_inference_features
from api.predictor import run_inference
from api.schemas import ForecastRequest, ForecastResponse, HourlyForecast

logger = logging.getLogger("nordspot.api.forecast")

router = APIRouter()


def _persist_predictions(preds: pd.DataFrame, date: str) -> None:
    """Write ens_q05/q50/q95 to TimeDB for monitoring.

    Called after successful inference.  Failures are logged but never
    propagate to the caller - a storage error must not break the API response.

    Parameters
    ----------
    preds:
        24-row DataFrame with columns ens_q05, ens_q50, ens_q95.
        Index must be a UTC DatetimeIndex aligned to the delivery hours.
    date:
        ISO date string (YYYY-MM-DD) for the delivery day, used only for
        log messages.
    """
    try:
        from db.schema import SERIES, init_schema
        from pipeline.store import write_series

        td = init_schema()
        for col, series_key in (
            ("ens_q05", "ens_q05"),
            ("ens_q50", "ens_q50"),
            ("ens_q95", "ens_q95"),
        ):
            s = preds[col].dropna()
            if s.empty:
                continue
            df_out = pd.DataFrame({"valid_time": s.index, "value": s.values})
            write_series(td, SERIES[series_key], df_out, retention="forever")

        logger.info("Persisted %d forecast rows to TimeDB for %s", len(preds), date)
    except Exception:
        logger.exception("Failed to persist forecast to TimeDB - continuing")


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

    # -- Persist to TimeDB for monitoring ----------------------------------
    _persist_predictions(preds, body.date)

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
