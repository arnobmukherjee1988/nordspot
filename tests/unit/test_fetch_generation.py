"""Unit tests for fetch_generation - mocks ENTSO-E, no real network calls."""

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from config.zone_config import load_zone
from pipeline.fetch_generation import fetch_generation


@pytest.fixture
def se3():
    return load_zone("SE3")


def _mock_generation_df(n_hours: int = 24) -> pd.DataFrame:
    """Flat-column generation DataFrame (post-MultiIndex normalisation)."""
    index = pd.date_range("2026-06-25", periods=n_hours, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "Wind Onshore": [500.0] * n_hours,
            "Solar": [100.0] * n_hours,
            "Nuclear": [3000.0] * n_hours,
        },
        index=index,
    )


def test_fetch_generation_returns_required_columns(se3):
    with patch("pipeline.fetch_generation.EntsoePandasClient") as mock_cls:
        mock_cls.return_value.query_generation.return_value = _mock_generation_df()
        df = fetch_generation(
            se3,
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert "valid_time" in df.columns
    assert "zone" in df.columns
    fuel_cols = [c for c in df.columns if c not in ("valid_time", "zone")]
    assert len(fuel_cols) >= 1


def test_fetch_generation_attaches_zone_eic(se3):
    with patch("pipeline.fetch_generation.EntsoePandasClient") as mock_cls:
        mock_cls.return_value.query_generation.return_value = _mock_generation_df()
        df = fetch_generation(
            se3,
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert (df["zone"] == "10Y1001A1001A46L").all()


def test_fetch_generation_returns_24_rows(se3):
    with patch("pipeline.fetch_generation.EntsoePandasClient") as mock_cls:
        mock_cls.return_value.query_generation.return_value = _mock_generation_df(24)
        df = fetch_generation(
            se3,
            start=datetime(2026, 6, 25, tzinfo=timezone.utc),
            end=datetime(2026, 6, 26, tzinfo=timezone.utc),
        )
    assert len(df) == 24
