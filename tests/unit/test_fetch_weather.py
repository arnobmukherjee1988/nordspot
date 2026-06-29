"""Unit tests for fetch_weather - mocks Open-Meteo, no real network calls."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.zone_config import load_zone
from pipeline.fetch_weather import fetch_weather


@pytest.fixture
def se3():
    return load_zone("SE3")


def _make_mock_client(n_hours: int = 24):
    """Build a mock openmeteo client returning n_hours of fake data."""
    times = pd.date_range("2026-06-25", periods=n_hours, freq="h", tz="UTC")

    mock_hourly = MagicMock()
    mock_hourly.Time.return_value = times[0].timestamp()
    mock_hourly.TimeEnd.return_value = times[-1].timestamp() + 3600
    mock_hourly.Interval.return_value = 3600
    mock_hourly.Variables.return_value.ValuesAsNumpy.return_value = np.full(
        n_hours, 10.0
    )

    mock_response = MagicMock()
    mock_response.Hourly.return_value = mock_hourly

    mock_client = MagicMock()
    mock_client.weather_api.return_value = [mock_response]
    return mock_client


def test_fetch_weather_returns_three_variables(se3):
    with patch(
        "pipeline.fetch_weather._build_client", return_value=_make_mock_client()
    ):
        result = fetch_weather(
            se3,
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert set(result.keys()) == {
        "temperature_2m",
        "wind_speed_10m",
        "shortwave_radiation",
    }


def test_fetch_weather_uses_zone_coordinates(se3):
    mock_client = _make_mock_client()
    with patch("pipeline.fetch_weather._build_client", return_value=mock_client):
        fetch_weather(
            se3,
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    call_params = mock_client.weather_api.call_args.kwargs["params"]
    assert call_params["latitude"] == se3.lat
    assert call_params["longitude"] == se3.lon
