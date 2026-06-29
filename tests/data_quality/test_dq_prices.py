"""Data-quality tests for the prices Bronze expectation suite.

Tests cover:
    - valid data passes all expectations
    - null values in ``value`` are flagged
    - duplicate timestamps are flagged
    - out-of-range prices are flagged
    - missing zone column is flagged
"""

from __future__ import annotations

import pandas as pd

from tests.data_quality.suite_prices import validate_prices


def _good_df(n: int = 24) -> pd.DataFrame:
    """A well-formed prices DataFrame that should pass all checks."""
    return pd.DataFrame(
        {
            "valid_time": pd.date_range("2026-06-25", periods=n, freq="h", tz="UTC"),
            "value": [50.0] * n,
            "zone": ["10Y1001A1001A46L"] * n,
        }
    )


def test_valid_prices_pass():
    gdf = validate_prices(_good_df())
    result = gdf.validate()
    assert result["success"] is True


def test_null_value_fails():
    df = _good_df()
    df.loc[0, "value"] = None
    gdf = validate_prices(df)
    result = gdf.validate()
    assert result["success"] is False


def test_duplicate_timestamps_fail():
    df = _good_df()
    df.loc[1, "valid_time"] = df.loc[0, "valid_time"]  # duplicate
    gdf = validate_prices(df)
    result = gdf.validate()
    assert result["success"] is False


def test_price_above_max_fails():
    df = _good_df()
    df.loc[0, "value"] = 10_000.0  # above 5 000 EUR/MWh cap
    gdf = validate_prices(df)
    result = gdf.validate()
    assert result["success"] is False


def test_price_below_min_fails():
    df = _good_df()
    df.loc[0, "value"] = -1_000.0  # below -500 EUR/MWh floor
    gdf = validate_prices(df)
    result = gdf.validate()
    assert result["success"] is False


def test_missing_zone_column_fails():
    df = _good_df().drop(columns=["zone"])
    gdf = validate_prices(df)
    result = gdf.validate()
    assert result["success"] is False
