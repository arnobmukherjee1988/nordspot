"""Unit tests for fetch_crossborder — mocks ENTSO-E, no real network calls."""

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from pipeline.fetch_crossborder import fetch_crossborder


def _mock_flow_series(n_hours: int = 24) -> pd.Series:
    """Simulate entsoe-py query_crossborder_flows output: hourly Series."""
    index = pd.date_range("2026-06-25", periods=n_hours, freq="h", tz="UTC")
    return pd.Series([500.0] * n_hours, index=index)


def test_fetch_crossborder_returns_correct_columns():
    with patch("pipeline.fetch_crossborder.EntsoePandasClient") as mock_cls:
        mock_cls.return_value.query_crossborder_flows.return_value = _mock_flow_series()
        df = fetch_crossborder(
            "SE3",
            "SE4",
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert set(df.columns) == {"valid_time", "value", "from_zone", "to_zone"}


def test_fetch_crossborder_attaches_correct_eics():
    with patch("pipeline.fetch_crossborder.EntsoePandasClient") as mock_cls:
        mock_cls.return_value.query_crossborder_flows.return_value = _mock_flow_series()
        df = fetch_crossborder(
            "SE3",
            "SE4",
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert (df["from_zone"] == "10Y1001A1001A46L").all()  # SE3
    assert (df["to_zone"] == "10Y1001A1001A47J").all()  # SE4


def test_fetch_crossborder_drops_nulls():
    index = pd.date_range("2026-06-25", periods=3, freq="h", tz="UTC")
    series_with_null = pd.Series([500.0, None, 600.0], index=index)
    with patch("pipeline.fetch_crossborder.EntsoePandasClient") as mock_cls:
        mock_cls.return_value.query_crossborder_flows.return_value = series_with_null
        df = fetch_crossborder(
            "SE3",
            "SE4",
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert len(df) == 2
