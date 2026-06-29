"""Pydantic request/response schemas for the NordSpot API.

All inputs are validated automatically by FastAPI before reaching the endpoint.
Invalid requests (wrong zone, past date) are rejected with HTTP 422 before
any model or database is touched.

Schemas:
    ForecastRequest  — POST /v1/forecast body
    HourlyForecast   — one row of the 24-hour output (hour + point + interval)
    ForecastResponse — full response envelope
"""

from __future__ import annotations

import datetime as dt
from datetime import date, datetime
from typing import List, Literal

from pydantic import BaseModel, field_validator


class ForecastRequest(BaseModel):
    """Input for a 24-hour ahead probabilistic price forecast.

    Attributes:
        zone: Swedish bidding zone identifier.
        date: The calendar date for which to forecast (must be tomorrow or later).
              Day-ahead prices are published the evening before delivery, so
              requesting today or yesterday has no operational value.
    """

    zone: Literal["SE1", "SE2", "SE3", "SE4"]
    date: date

    @field_validator("date")
    @classmethod
    def date_must_be_future(cls, v: date) -> date:
        tomorrow = dt.date.today() + dt.timedelta(days=1)
        if v < tomorrow:
            raise ValueError(f"date must be tomorrow ({tomorrow}) or later; got {v}")
        return v


class HourlyForecast(BaseModel):
    """Probabilistic forecast for a single delivery hour.

    Attributes:
        hour:  Delivery hour (0-23) in local time (Europe/Stockholm).
        point: Median (q50) forecast in EUR/MWh.
        q05:   5th percentile — lower bound of the 90% prediction interval.
        q95:   95th percentile — upper bound of the 90% prediction interval.
    """

    hour: int
    point: float
    q05: float
    q95: float


class ForecastResponse(BaseModel):
    """Full 24-hour forecast response envelope.

    Attributes:
        zone:          Bidding zone (mirrors request).
        date:          Delivery date (mirrors request).
        model_version: MLflow model registry version that generated the forecast.
        generated_at:  UTC timestamp when the forecast was produced.
        forecast:      List of 24 HourlyForecast objects (hour 0 through 23).
    """

    zone: str
    date: date
    model_version: str
    generated_at: datetime
    forecast: List[HourlyForecast]
