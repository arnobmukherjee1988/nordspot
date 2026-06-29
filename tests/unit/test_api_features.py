"""Unit tests for api/features.py - inference feature retrieval.

All tests run without a real database.  build_features() and load_zone()
are patched at their source modules so no TimeDB or ClickHouse connections
are made.  The mock DataFrames mirror the shape that pipeline.features
.build_features() produces.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from api.features import get_inference_features

# -- Constants -----------------------------------------------------------------

_ZONE_NAME = "SE3"
_ZONE_EIC = "10Y1001A1001A46L"
_TARGET_DATE = datetime.date(2026, 7, 1)

# -- Helpers -------------------------------------------------------------------


def _make_df(n_days: int = 3, zone_eic: str = _ZONE_EIC) -> pd.DataFrame:
    """Synthetic DataFrame mirroring build_features() output.

    n_days=3 starting 2026-06-29 covers:
        day 0: 2026-06-29 (24 rows)
        day 1: 2026-06-30 (24 rows)
        day 2: 2026-07-01 (24 rows) <- _TARGET_DATE
    """
    start = pd.Timestamp("2026-06-29T00:00:00Z")
    idx = pd.date_range(start, periods=n_days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    n = len(idx)
    return pd.DataFrame(
        {
            "valid_time": idx,
            "zone": zone_eic,
            "price": rng.normal(50, 10, n),
            "price_lag24h": rng.normal(50, 10, n),
            "price_lag168h": rng.normal(50, 10, n),
            "price_roll24h_mean": rng.normal(50, 5, n),
            "hour": [t.hour for t in idx],
            "weekday": [t.dayofweek for t in idx],
            "temperature_2m": rng.normal(15, 5, n),
            "wind_speed_10m": rng.uniform(0, 15, n),
            "load_mw": rng.normal(5000, 500, n),
        }
    )


def _make_df_missing_date(zone_eic: str = _ZONE_EIC) -> pd.DataFrame:
    """DataFrame that does NOT contain any rows for _TARGET_DATE."""
    start = pd.Timestamp("2026-06-29T00:00:00Z")
    idx = pd.date_range(start, periods=2 * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "valid_time": idx,
            "zone": zone_eic,
            "price": rng.normal(50, 10, len(idx)),
        }
    )


def _make_df_partial_date(zone_eic: str = _ZONE_EIC) -> pd.DataFrame:
    """DataFrame where _TARGET_DATE has only 23 rows (one hour missing)."""
    df = _make_df(zone_eic=zone_eic)
    target_rows = df[df["valid_time"].dt.date == _TARGET_DATE]
    drop_idx = target_rows.index[0]  # drop the first hour of target date
    return df.drop(index=drop_idx).reset_index(drop=True)


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def mock_zone():
    zone = MagicMock()
    zone.entsoe_eic = _ZONE_EIC
    zone.name = "Malmö"
    return zone


# -- Happy path ----------------------------------------------------------------


@patch("pipeline.features.build_features")
@patch("config.zone_config.load_zone")
def test_returns_24_rows(mock_load_zone, mock_build, mock_zone):
    mock_load_zone.return_value = mock_zone
    mock_build.return_value = _make_df()

    result = get_inference_features(_ZONE_NAME, _TARGET_DATE)

    assert len(result) == 24


@patch("pipeline.features.build_features")
@patch("config.zone_config.load_zone")
def test_all_valid_times_are_target_date(mock_load_zone, mock_build, mock_zone):
    mock_load_zone.return_value = mock_zone
    mock_build.return_value = _make_df()

    result = get_inference_features(_ZONE_NAME, _TARGET_DATE)

    dates = result["valid_time"].dt.date.unique()
    assert list(dates) == [_TARGET_DATE]


@patch("pipeline.features.build_features")
@patch("config.zone_config.load_zone")
def test_hours_are_0_to_23_in_order(mock_load_zone, mock_build, mock_zone):
    mock_load_zone.return_value = mock_zone
    mock_build.return_value = _make_df()

    result = get_inference_features(_ZONE_NAME, _TARGET_DATE)

    assert list(result["valid_time"].dt.hour) == list(range(24))


@patch("pipeline.features.build_features")
@patch("config.zone_config.load_zone")
def test_zone_column_is_present(mock_load_zone, mock_build, mock_zone):
    mock_load_zone.return_value = mock_zone
    mock_build.return_value = _make_df()

    result = get_inference_features(_ZONE_NAME, _TARGET_DATE)

    assert "zone" in result.columns


@patch("pipeline.features.build_features")
@patch("config.zone_config.load_zone")
def test_valid_time_column_is_present(mock_load_zone, mock_build, mock_zone):
    mock_load_zone.return_value = mock_zone
    mock_build.return_value = _make_df()

    result = get_inference_features(_ZONE_NAME, _TARGET_DATE)

    assert "valid_time" in result.columns


@patch("pipeline.features.build_features")
@patch("config.zone_config.load_zone")
def test_calls_load_zone_with_correct_name(mock_load_zone, mock_build, mock_zone):
    mock_load_zone.return_value = mock_zone
    mock_build.return_value = _make_df()

    get_inference_features(_ZONE_NAME, _TARGET_DATE)

    mock_load_zone.assert_called_once_with(_ZONE_NAME)


@patch("pipeline.features.build_features")
@patch("config.zone_config.load_zone")
def test_result_index_is_reset(mock_load_zone, mock_build, mock_zone):
    mock_load_zone.return_value = mock_zone
    mock_build.return_value = _make_df()

    result = get_inference_features(_ZONE_NAME, _TARGET_DATE)

    assert list(result.index) == list(range(24))


# -- Error paths ---------------------------------------------------------------


@patch("pipeline.features.build_features")
@patch("config.zone_config.load_zone")
def test_raises_when_no_data_for_date(mock_load_zone, mock_build, mock_zone):
    mock_load_zone.return_value = mock_zone
    mock_build.return_value = _make_df_missing_date()

    with pytest.raises(ValueError, match=str(_TARGET_DATE)):
        get_inference_features(_ZONE_NAME, _TARGET_DATE)


@patch("pipeline.features.build_features")
@patch("config.zone_config.load_zone")
def test_raises_when_fewer_than_24_rows(mock_load_zone, mock_build, mock_zone):
    mock_load_zone.return_value = mock_zone
    mock_build.return_value = _make_df_partial_date()

    with pytest.raises(ValueError, match="24"):
        get_inference_features(_ZONE_NAME, _TARGET_DATE)
