"""Unit tests for fetch_load — mocks ENTSO-E, no real network calls."""

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from config.zone_config import load_zone
from pipeline.fetch_load import fetch_load


@pytest.fixture
def se3():
    return load_zone("SE3")


def _mock_load_df(n_hours: int = 24) -> pd.DataFrame:
    """Simulate entsoe-py query_load output: DatetimeIndex + 'Actual Load' column."""
    index = pd.date_range("2026-06-25", periods=n_hours, freq="h", tz="UTC")
    return pd.DataFrame({"Actual Load": [12000.0] * n_hours}, index=index)


def test_fetch_load_returns_correct_columns(se3):
    with patch("pipeline.fetch_load.EntsoePandasClient") as mock_cls:
        mock_cls.return_value.query_load.return_value = _mock_load_df()
        df = fetch_load(
            se3,
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert set(df.columns) == {"valid_time", "actual_load", "zone"}


def test_fetch_load_attaches_zone_eic(se3):
    with patch("pipeline.fetch_load.EntsoePandasClient") as mock_cls:
        mock_cls.return_value.query_load.return_value = _mock_load_df()
        df = fetch_load(
            se3,
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert (df["zone"] == "10Y1001A1001A46L").all()


def test_fetch_load_drops_nulls(se3):
    index = pd.date_range("2026-06-25", periods=3, freq="h", tz="UTC")
    df_with_null = pd.DataFrame({"Actual Load": [12000.0, None, 11500.0]}, index=index)
    with patch("pipeline.fetch_load.EntsoePandasClient") as mock_cls:
        mock_cls.return_value.query_load.return_value = df_with_null
        df = fetch_load(
            se3,
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert len(df) == 2
