"""Unit tests for POST /v1/forecast.

Story 5.4: the endpoint now requires a loaded Production model (503 otherwise)
and calls get_inference_features() + run_inference() for real predictions.

All tests use an autouse fixture that:
    - injects a ready ModelStore so happy-path requests get 200
    - stubs get_inference_features and run_inference so no DB or model
      files are accessed

Tests cover:
    - Happy path: valid zone + future date → 200 with 24-hour forecast
    - Input validation: invalid zone, today, yesterday → 422
    - Response shape: 24 hours, hours 0-23, q05/point/q95 present
    - Zone mirroring: response zone matches request zone
    - Service unavailable: no Production model → 503
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.model_store import ModelStore

client = TestClient(app)

_TOMORROW = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
_TODAY = datetime.date.today().isoformat()
_YESTERDAY = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
_NEXT_WEEK = (datetime.date.today() + datetime.timedelta(days=7)).isoformat()

# ── Stub prediction data ──────────────────────────────────────────────────────

_STUB_FEATURES = pd.DataFrame({"dummy": range(24)})
_STUB_PREDS = pd.DataFrame(
    {
        "ens_q05": [30.0] * 24,
        "ens_q50": [50.0] * 24,
        "ens_q95": [70.0] * 24,
    }
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _ready_store_and_mock_inference(monkeypatch):
    """Inject a ready model store and stub the data/inference pipeline.

    Applied to every test in this module so no DB connections or model
    files are needed.  The 503 test overrides app.state.model_store
    within the test body.
    """
    app.state.model_store = ModelStore(model=MagicMock(), model_version="test-v1")
    monkeypatch.setattr(
        "api.routers.forecast.get_inference_features",
        lambda *a, **kw: _STUB_FEATURES,
    )
    monkeypatch.setattr(
        "api.routers.forecast.run_inference",
        lambda *a, **kw: _STUB_PREDS,
    )
    yield
    # Restore default not-ready store after each test
    app.state.model_store = ModelStore()


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


def test_forecast_point_comes_from_ens_q50():
    response = client.post("/v1/forecast", json={"zone": "SE3", "date": _TOMORROW})
    assert response.json()["forecast"][0]["point"] == 50.0


def test_forecast_q05_comes_from_ens_q05():
    response = client.post("/v1/forecast", json={"zone": "SE3", "date": _TOMORROW})
    assert response.json()["forecast"][0]["q05"] == 30.0


def test_forecast_q95_comes_from_ens_q95():
    response = client.post("/v1/forecast", json={"zone": "SE3", "date": _TOMORROW})
    assert response.json()["forecast"][0]["q95"] == 70.0


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


# ── Service unavailable (503) ─────────────────────────────────────────────────


def test_forecast_returns_503_when_model_not_loaded():
    # Override the ready store injected by the autouse fixture
    app.state.model_store = ModelStore()  # not ready
    response = client.post("/v1/forecast", json={"zone": "SE3", "date": _TOMORROW})
    assert response.status_code == 503
