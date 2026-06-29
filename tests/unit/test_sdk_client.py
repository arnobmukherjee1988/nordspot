"""Unit tests for sdk/nordspot_client.py.

All HTTP calls are intercepted by patching requests.Session.send so no
real network connections are made.  Each test builds a minimal mock
Response and verifies the client's behaviour.
"""

from __future__ import annotations

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from sdk import NordSpotClient, NordSpotError

# -- Constants -----------------------------------------------------------------

_BASE_URL = "http://testserver"
_API_KEY = "test-key-xyz"
_TOMORROW = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

_HEALTH_PAYLOAD = {
    "status": "ok",
    "model_version": "5",
    "trained_at": "2026-06-28T12:00:00Z",
    "zones_covered": ["SE1", "SE2", "SE3", "SE4"],
    "api_version": "v1",
    "timestamp": "2026-06-30T08:00:00Z",
}

_FORECAST_PAYLOAD = {
    "zone": "SE3",
    "date": _TOMORROW,
    "model_version": "5",
    "generated_at": "2026-06-30T08:00:00Z",
    "forecast": [
        {"hour": h, "point": 60.0, "q05": 45.0, "q95": 78.0} for h in range(24)
    ],
}

# -- Helpers -------------------------------------------------------------------


def _mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    """Build a minimal mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


@pytest.fixture
def client():
    return NordSpotClient(base_url=_BASE_URL, api_key=_API_KEY)


# -- health() tests ------------------------------------------------------------


def test_health_returns_dict(client):
    with patch.object(
        client._session, "get", return_value=_mock_response(_HEALTH_PAYLOAD)
    ):
        result = client.health()
    assert isinstance(result, dict)


def test_health_returns_status_ok(client):
    with patch.object(
        client._session, "get", return_value=_mock_response(_HEALTH_PAYLOAD)
    ):
        result = client.health()
    assert result["status"] == "ok"


def test_health_hits_correct_url(client):
    with patch.object(
        client._session, "get", return_value=_mock_response(_HEALTH_PAYLOAD)
    ) as mock_get:
        client.health()
    mock_get.assert_called_once_with(f"{_BASE_URL}/health", timeout=30)


# -- get_forecast() tests ------------------------------------------------------


def test_get_forecast_returns_dict(client):
    with patch.object(
        client._session, "post", return_value=_mock_response(_FORECAST_PAYLOAD)
    ):
        result = client.get_forecast(zone="SE3", date=_TOMORROW)
    assert isinstance(result, dict)


def test_get_forecast_has_24_hours(client):
    with patch.object(
        client._session, "post", return_value=_mock_response(_FORECAST_PAYLOAD)
    ):
        result = client.get_forecast(zone="SE3", date=_TOMORROW)
    assert len(result["forecast"]) == 24


def test_get_forecast_hits_correct_url(client):
    with patch.object(
        client._session, "post", return_value=_mock_response(_FORECAST_PAYLOAD)
    ) as mock_post:
        client.get_forecast(zone="SE3", date=_TOMORROW)
    args, kwargs = mock_post.call_args
    assert args[0] == f"{_BASE_URL}/v1/forecast"


def test_get_forecast_sends_zone_and_date(client):
    with patch.object(
        client._session, "post", return_value=_mock_response(_FORECAST_PAYLOAD)
    ) as mock_post:
        client.get_forecast(zone="SE3", date=_TOMORROW)
    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {"zone": "SE3", "date": _TOMORROW}


def test_get_forecast_accepts_date_object(client):
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    with patch.object(
        client._session, "post", return_value=_mock_response(_FORECAST_PAYLOAD)
    ) as mock_post:
        client.get_forecast(zone="SE3", date=tomorrow)
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["date"] == tomorrow.isoformat()


def test_api_key_header_sent(client):
    """X-API-Key must be present on the session's default headers."""
    assert client._session.headers["X-API-Key"] == _API_KEY


# -- Error handling ------------------------------------------------------------


def test_non_2xx_raises_nordspot_error(client):
    error_payload = {"detail": "Invalid or missing API key."}
    with patch.object(
        client._session,
        "post",
        return_value=_mock_response(error_payload, status_code=401),
    ):
        with pytest.raises(NordSpotError):
            client.get_forecast(zone="SE3", date=_TOMORROW)


def test_nordspot_error_carries_status_code(client):
    error_payload = {"detail": "No Production model is loaded."}
    with patch.object(
        client._session,
        "post",
        return_value=_mock_response(error_payload, status_code=503),
    ):
        with pytest.raises(NordSpotError) as exc_info:
            client.get_forecast(zone="SE3", date=_TOMORROW)
    assert exc_info.value.status_code == 503


def test_nordspot_error_carries_detail(client):
    error_payload = {"detail": "No Production model is loaded."}
    with patch.object(
        client._session,
        "post",
        return_value=_mock_response(error_payload, status_code=503),
    ):
        with pytest.raises(NordSpotError) as exc_info:
            client.get_forecast(zone="SE3", date=_TOMORROW)
    assert "No Production model" in exc_info.value.detail


def test_nordspot_error_str_contains_status_and_detail(client):
    error_payload = {"detail": "Not found."}
    with patch.object(
        client._session,
        "post",
        return_value=_mock_response(error_payload, status_code=404),
    ):
        with pytest.raises(NordSpotError) as exc_info:
            client.get_forecast(zone="SE3", date=_TOMORROW)
    err_str = str(exc_info.value)
    assert "404" in err_str
    assert "Not found" in err_str


# -- Miscellaneous -------------------------------------------------------------


def test_base_url_trailing_slash_stripped():
    c = NordSpotClient(base_url="http://example.com/", api_key="k")
    assert c._base_url == "http://example.com"


def test_custom_timeout_stored():
    c = NordSpotClient(base_url=_BASE_URL, api_key=_API_KEY, timeout=10)
    assert c._timeout == 10
