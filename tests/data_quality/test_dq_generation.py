"""Data-quality tests for the generation Bronze expectation suite.

Tests cover:
    - valid wide DataFrame passes all expectations
    - null valid_time is flagged
    - negative MW value in a fuel column is flagged
    - missing zone column is flagged
    - DataFrame with no fuel columns (only mandatory cols) is flagged
"""

from __future__ import annotations

import pandas as pd

from tests.data_quality.suite_generation import validate_generation


def _good_df(n: int = 24) -> pd.DataFrame:
    """A well-formed generation DataFrame that should pass all checks."""
    return pd.DataFrame(
        {
            "valid_time": pd.date_range("2026-06-25", periods=n, freq="h", tz="UTC"),
            "zone": ["10Y1001A1001A46L"] * n,
            "Wind Onshore": [500.0] * n,
            "Solar": [100.0] * n,
            "Nuclear": [3_000.0] * n,
        }
    )


def test_valid_generation_passes():
    gdf = validate_generation(_good_df())
    result = gdf.validate()
    assert result["success"] is True


def test_null_valid_time_fails():
    df = _good_df()
    df.loc[0, "valid_time"] = None
    gdf = validate_generation(df)
    result = gdf.validate()
    assert result["success"] is False


def test_negative_mw_fails():
    df = _good_df()
    df.loc[0, "Wind Onshore"] = -50.0  # generation cannot be negative
    gdf = validate_generation(df)
    result = gdf.validate()
    assert result["success"] is False


def test_missing_zone_column_fails():
    df = _good_df().drop(columns=["zone"])
    gdf = validate_generation(df)
    result = gdf.validate()
    assert result["success"] is False


def test_no_fuel_columns_fails():
    """A DataFrame with only valid_time + zone (no fuel data) must fail."""
    df = _good_df()[["valid_time", "zone"]]
    gdf = validate_generation(df)
    result = gdf.validate()
    assert result["success"] is False
