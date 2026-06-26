"""Unit tests for fetch_prices — mocks the ENTSO-E API, no real network calls."""

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from config.zone_config import load_zone
from pipeline.fetch_prices import fetch_prices


@pytest.fixture
def se3():
    return load_zone("SE3")


@pytest.fixture
def mock_entsoe_series():
    """Fake ENTSO-E response: 24 hourly prices on 2026-06-25."""
    index = pd.date_range("2026-06-25", periods=24, freq="h", tz="UTC")
    return pd.Series([55.0] * 24, index=index)


def test_fetch_prices_returns_correct_columns(se3, mock_entsoe_series):
    with patch("pipeline.fetch_prices.EntsoePandasClient") as mock_client:
        mock_client.return_value.query_day_ahead_prices.return_value = (
            mock_entsoe_series
        )
        df = fetch_prices(
            se3,
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert set(df.columns) == {"valid_time", "value", "zone"}
    assert len(df) == 24


def test_fetch_prices_attaches_zone_eic(se3, mock_entsoe_series):
    with patch("pipeline.fetch_prices.EntsoePandasClient") as mock_client:
        mock_client.return_value.query_day_ahead_prices.return_value = (
            mock_entsoe_series
        )
        df = fetch_prices(
            se3,
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert (df["zone"] == "10Y1001A1001A46L").all()


def test_fetch_prices_drops_nulls(se3):
    index = pd.date_range("2026-06-25", periods=3, freq="h", tz="UTC")
    series_with_nulls = pd.Series([55.0, None, 60.0], index=index)
    with patch("pipeline.fetch_prices.EntsoePandasClient") as mock_client:
        mock_client.return_value.query_day_ahead_prices.return_value = series_with_nulls
        df = fetch_prices(
            se3,
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert len(df) == 2
