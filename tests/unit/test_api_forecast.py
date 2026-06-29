"""Unit tests for POST /v1/forecast.

Tests cover:
    - Happy path: valid zone + future date → 200 with 24-hour forecast
    - Input validation: invalid zone, today, yesterday → 422
    - Response shape: 24 hours, hours 0-23, q05/point/q95 present
    - Zone mirroring: response zone matches request zone

Story 5.1 stub returns synthetic flat data (50 EUR/MWh).
Tests remain valid after Stories 5.2-5.4 replace stub with real predictions,
because they check structure and validation, not specific values.
"""

from __future__ import annotations

import datetime

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

_TOMORROW = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
_TODAY = datetime.date.today().isoformat()
_YESTERDAY = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
_NEXT_WEEK = (datetime.date.today() + datetime.timedelta(days=7)).isoformat()


# ── Happy path ────────────────────────────────────────────────────────────────


def test_forecast_returns_200():
    response = client.post("/v1/forecast", json={"zone": "SE3", "date": _TOMORROW})
    assert response.status_code == 200


def test_forecast_returns_24_hours():
    response = client.post("/v1/forecast", json={"zone": "SE3", "date": _TOMORROW})
    assert len(response.json()["forecast"]) == 24


def test_forecast_hours_are_0_to_23():
    response = client.post("/v1/forecast", json={"zone": "SE3", "date": _TOMORROW})
    hours = [h["hour"] for h in response.json()["forecast"]]
    assert hours == list(range(24))


def test_forecast_each_hour_has_point_q05_q95():
    response = client.post("/v1/forecast", json={"zone": "SE3", "date": _TOMORROW})
    for hour_fc in response.json()["forecast"]:
        assert "point" in hour_fc
        assert "q05" in hour_fc
        assert "q95" in hour_fc


def test_forecast_zone_mirrors_request():
    response = client.post("/v1/forecast", json={"zone": "SE1", "date": _TOMORROW})
    assert response.json()["zone"] == "SE1"


def test_forecast_date_mirrors_request():
    response = client.post("/v1/forecast", json={"zone": "SE2", "date": _TOMORROW})
    assert response.json()["date"] == _TOMORROW


def test_forecast_has_model_version():
    response = client.post("/v1/forecast", json={"zone": "SE3", "date": _TOMORROW})
    assert "model_version" in response.json()


def test_forecast_has_generated_at():
    response = client.post("/v1/forecast", json={"zone": "SE3", "date": _TOMORROW})
    assert "generated_at" in response.json()


def test_forecast_future_date_accepted():
    response = client.post("/v1/forecast", json={"zone": "SE4", "date": _NEXT_WEEK})
    assert response.status_code == 200


def test_forecast_all_zones_accepted():
    for zone in ["SE1", "SE2", "SE3", "SE4"]:
        response = client.post("/v1/forecast", json={"zone": zone, "date": _TOMORROW})
        assert response.status_code == 200, f"Zone {zone} failed"


# ── Validation errors (422) ───────────────────────────────────────────────────


def test_forecast_invalid_zone_returns_422():
    response = client.post("/v1/forecast", json={"zone": "SE5", "date": _TOMORROW})
    assert response.status_code == 422


def test_forecast_today_returns_422():
    response = client.post("/v1/forecast", json={"zone": "SE3", "date": _TODAY})
    assert response.status_code == 422


def test_forecast_past_date_returns_422():
    response = client.post("/v1/forecast", json={"zone": "SE3", "date": _YESTERDAY})
    assert response.status_code == 422


def test_forecast_missing_zone_returns_422():
    response = client.post("/v1/forecast", json={"date": _TOMORROW})
    assert response.status_code == 422


def test_forecast_missing_date_returns_422():
    response = client.post("/v1/forecast", json={"zone": "SE3"})
    assert response.status_code == 422


def test_forecast_empty_body_returns_422():
    response = client.post("/v1/forecast", json={})
    assert response.status_code == 422
