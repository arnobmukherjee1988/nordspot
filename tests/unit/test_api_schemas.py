"""Unit tests for api/schemas.py.

Verifies Pydantic validation for ForecastRequest, HourlyForecast, and
ForecastResponse without starting the API server.

Key validations:
    - Only SE1/SE2/SE3/SE4 are valid zones
    - date must be tomorrow or later (today and past dates rejected)
    - Response schema round-trips cleanly
"""

from __future__ import annotations

import datetime

import pytest
from pydantic import ValidationError

from api.schemas import ForecastRequest, ForecastResponse, HourlyForecast

_TOMORROW = datetime.date.today() + datetime.timedelta(days=1)
_TODAY = datetime.date.today()
_YESTERDAY = datetime.date.today() - datetime.timedelta(days=1)
_NEXT_WEEK = datetime.date.today() + datetime.timedelta(days=7)


# ── ForecastRequest ───────────────────────────────────────────────────────────


def test_forecast_request_valid():
    req = ForecastRequest(zone="SE3", date=_TOMORROW)
    assert req.zone == "SE3"
    assert req.date == _TOMORROW


def test_forecast_request_all_zones_valid():
    for zone in ["SE1", "SE2", "SE3", "SE4"]:
        req = ForecastRequest(zone=zone, date=_TOMORROW)
        assert req.zone == zone


def test_forecast_request_future_date_valid():
    req = ForecastRequest(zone="SE1", date=_NEXT_WEEK)
    assert req.date == _NEXT_WEEK


def test_forecast_request_invalid_zone_raises():
    with pytest.raises(ValidationError):
        ForecastRequest(zone="SE5", date=_TOMORROW)


def test_forecast_request_lowercase_zone_raises():
    with pytest.raises(ValidationError):
        ForecastRequest(zone="se3", date=_TOMORROW)


def test_forecast_request_today_raises():
    """Today is not a valid forecast date — day-ahead requires at least tomorrow."""
    with pytest.raises(ValidationError):
        ForecastRequest(zone="SE3", date=_TODAY)


def test_forecast_request_past_date_raises():
    with pytest.raises(ValidationError):
        ForecastRequest(zone="SE3", date=_YESTERDAY)


# ── HourlyForecast ────────────────────────────────────────────────────────────


def test_hourly_forecast_fields():
    hf = HourlyForecast(hour=12, point=55.0, q05=30.0, q95=80.0)
    assert hf.hour == 12
    assert hf.point == 55.0
    assert hf.q05 == 30.0
    assert hf.q95 == 80.0


def test_hourly_forecast_hour_zero():
    hf = HourlyForecast(hour=0, point=40.0, q05=20.0, q95=60.0)
    assert hf.hour == 0


def test_hourly_forecast_hour_23():
    hf = HourlyForecast(hour=23, point=45.0, q05=25.0, q95=65.0)
    assert hf.hour == 23


# ── ForecastResponse ──────────────────────────────────────────────────────────


def test_forecast_response_round_trip():
    hours = [HourlyForecast(hour=h, point=50.0, q05=30.0, q95=70.0) for h in range(24)]
    resp = ForecastResponse(
        zone="SE3",
        date=_TOMORROW,
        model_version="v1",
        generated_at=datetime.datetime.now(datetime.timezone.utc),
        forecast=hours,
    )
    assert resp.zone == "SE3"
    assert resp.date == _TOMORROW
    assert len(resp.forecast) == 24


def test_forecast_response_forecast_length():
    hours = [HourlyForecast(hour=h, point=50.0, q05=30.0, q95=70.0) for h in range(24)]
    resp = ForecastResponse(
        zone="SE2",
        date=_TOMORROW,
        model_version="stub-5.1",
        generated_at=datetime.datetime.now(datetime.timezone.utc),
        forecast=hours,
    )
    assert len(resp.forecast) == 24


def test_forecast_response_serialises_to_dict():
    hours = [HourlyForecast(hour=h, point=50.0, q05=30.0, q95=70.0) for h in range(24)]
    resp = ForecastResponse(
        zone="SE4",
        date=_TOMORROW,
        model_version="stub-5.1",
        generated_at=datetime.datetime(2026, 6, 29, 12, 0, 0),
        forecast=hours,
    )
    d = resp.model_dump()
    assert d["zone"] == "SE4"
    assert len(d["forecast"]) == 24
    assert d["forecast"][0]["hour"] == 0
