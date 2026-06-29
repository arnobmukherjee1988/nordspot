"""Data-quality tests for the weather Bronze expectation suite.

Tests cover:
    - valid data passes all expectations
    - null temperature is flagged
    - temperature out of physical range is flagged
    - negative wind speed is flagged
    - negative shortwave radiation is flagged
    - missing column is flagged
"""

from __future__ import annotations

import pandas as pd

from tests.data_quality.suite_weather import validate_weather


def _good_df(n: int = 24) -> pd.DataFrame:
    """A well-formed weather DataFrame that should pass all checks."""
    return pd.DataFrame(
        {
            "valid_time": pd.date_range("2026-06-25", periods=n, freq="h", tz="UTC"),
            "temperature_2m": [15.0] * n,
            "wind_speed_10m": [5.0] * n,
            "shortwave_radiation": [200.0] * n,
            "zone": ["10Y1001A1001A46L"] * n,
        }
    )


def test_valid_weather_passes():
    gdf = validate_weather(_good_df())
    result = gdf.validate()
    assert result["success"] is True


def test_null_temperature_fails():
    df = _good_df()
    df.loc[0, "temperature_2m"] = None
    gdf = validate_weather(df)
    result = gdf.validate()
    assert result["success"] is False


def test_temperature_too_high_fails():
    df = _good_df()
    df.loc[0, "temperature_2m"] = 75.0  # above 60degC cap
    gdf = validate_weather(df)
    result = gdf.validate()
    assert result["success"] is False


def test_temperature_too_low_fails():
    df = _good_df()
    df.loc[0, "temperature_2m"] = -60.0  # below -50degC floor
    gdf = validate_weather(df)
    result = gdf.validate()
    assert result["success"] is False


def test_negative_wind_speed_fails():
    df = _good_df()
    df.loc[0, "wind_speed_10m"] = -1.0
    gdf = validate_weather(df)
    result = gdf.validate()
    assert result["success"] is False


def test_negative_radiation_fails():
    df = _good_df()
    df.loc[0, "shortwave_radiation"] = -10.0
    gdf = validate_weather(df)
    result = gdf.validate()
    assert result["success"] is False


def test_missing_column_fails():
    df = _good_df().drop(columns=["shortwave_radiation"])
    gdf = validate_weather(df)
    result = gdf.validate()
    assert result["success"] is False
