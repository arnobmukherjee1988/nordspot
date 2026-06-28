"""Unit tests for pipeline/features.py.

Strategy:
- Pure transformation helpers (_build_price_lags, _build_calendar, etc.) are
  tested directly with synthetic pandas objects — no DB required.
- build_features() is tested with mocked TD and ClickHouse clients injected
  via the td= and ch_client= parameters.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from config.zone_config import load_zone
from pipeline.features import (
    _build_calendar,
    _build_price_lags,
    _build_rolling_stats,
    _build_weather_interactions,
    _read_generation,
    _read_load,
    build_features,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

N_HOURS = 400  # enough to cover 168h+ lags without all-NaN tail


@pytest.fixture
def se3():
    return load_zone("SE3")


@pytest.fixture
def idx():
    return pd.date_range("2026-01-01", periods=N_HOURS, freq="h", tz="UTC")


@pytest.fixture
def price(idx):
    rng = np.random.default_rng(42)
    return pd.Series(rng.uniform(20, 200, N_HOURS), index=idx, name="price")


# ── Price lags ────────────────────────────────────────────────────────────────


def test_price_lags_returns_seven_series(price):
    lags = _build_price_lags(price)
    assert len(lags) == 7


def test_price_lag_24h_equals_shift(price):
    lags = {s.name: s for s in _build_price_lags(price)}
    pd.testing.assert_series_equal(
        lags["price_lag24h"],
        price.shift(24).rename("price_lag24h"),
    )


def test_price_lag_168h_equals_shift(price):
    lags = {s.name: s for s in _build_price_lags(price)}
    pd.testing.assert_series_equal(
        lags["price_lag168h"],
        price.shift(168).rename("price_lag168h"),
    )


def test_price_lags_are_non_negative_shift(price):
    """Lags must shift into the past — no negative (future) shifts."""
    for s in _build_price_lags(price):
        hours = int(s.name.replace("price_lag", "").replace("h", ""))
        assert hours > 0, f"Negative lag detected: {s.name}"


# ── Rolling stats ─────────────────────────────────────────────────────────────


def test_rolling_stats_returns_four_series(price):
    assert len(_build_rolling_stats(price)) == 4


def test_rolling_mean_24h_is_shifted(price):
    """Rolling mean must be computed on price.shift(24) to prevent leakage."""
    stats = {s.name: s for s in _build_rolling_stats(price)}
    expected = price.shift(24).rename("price").rolling(24, min_periods=1).mean()
    pd.testing.assert_series_equal(
        stats["price_roll24h_mean"],
        expected.rename("price_roll24h_mean"),
    )


# ── Calendar ──────────────────────────────────────────────────────────────────


def test_calendar_hour_range(idx):
    cal = _build_calendar(idx)
    assert cal["hour"].min() >= 0
    assert cal["hour"].max() <= 23


def test_calendar_weekday_range(idx):
    cal = _build_calendar(idx)
    assert cal["weekday"].min() >= 0
    assert cal["weekday"].max() <= 6


def test_calendar_cyclical_encoding_bounds(idx):
    cal = _build_calendar(idx)
    for col in (
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
    ):
        assert cal[col].min() >= -1.0 - 1e-9
        assert cal[col].max() <= 1.0 + 1e-9


def test_calendar_is_weekend_binary(idx):
    cal = _build_calendar(idx)
    assert set(cal["is_weekend"].unique()).issubset({0, 1})


def test_calendar_hour_x_month_interaction(idx):
    cal = _build_calendar(idx)
    expected = cal["hour"] * cal["month"]
    pd.testing.assert_series_equal(cal["hour_x_month"], expected, check_names=False)


# ── Weather interactions ──────────────────────────────────────────────────────


def test_weather_interactions_count(idx):
    temp = pd.Series(np.full(len(idx), 5.0), index=idx)
    wind = pd.Series(np.full(len(idx), 3.0), index=idx)
    hour = pd.Series(np.arange(len(idx)) % 24, index=idx)
    feats = _build_weather_interactions(temp, wind, hour)
    assert len(feats) == 2


def test_temp_x_wind_values(idx):
    temp = pd.Series(np.full(len(idx), 5.0), index=idx)
    wind = pd.Series(np.full(len(idx), 3.0), index=idx)
    hour = pd.Series(np.zeros(len(idx)), index=idx)
    feats = {s.name: s for s in _build_weather_interactions(temp, wind, hour)}
    assert (feats["temp_x_wind"] == 15.0).all()


# ── ClickHouse reader helpers (mocked) ────────────────────────────────────────


def _mock_ch(sql_to_df: dict[str, pd.DataFrame]) -> MagicMock:
    """Build a mock ch_client whose query_df returns different DFs by SQL snippet."""
    ch = MagicMock()

    def _query_df(sql):
        for key, df in sql_to_df.items():
            if key in sql:
                return df
        return pd.DataFrame()

    ch.query_df.side_effect = _query_df
    return ch


def _gen_df(idx):
    return pd.DataFrame({"valid_time": idx, "wind_mw": 500.0, "solar_mw": 100.0})


def _load_df(idx):
    return pd.DataFrame({"valid_time": idx, "load_mw": 12_000.0})


def test_read_generation_returns_correct_columns():
    idx = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    ch = _mock_ch({"silver_generation": _gen_df(idx)})
    df = _read_generation(
        ch,
        "10Y1001A1001A46L",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    assert "wind_mw" in df.columns
    assert "solar_mw" in df.columns


def test_read_load_returns_load_mw():
    idx = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    ch = _mock_ch({"silver_load": _load_df(idx)})
    df = _read_load(
        ch,
        "10Y1001A1001A46L",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    assert "load_mw" in df.columns


# ── build_features integration (mocked DB) ───────────────────────────────────


def _make_td_mock(idx):
    """TimeDBClient mock that returns synthetic price + weather series."""
    rng = np.random.default_rng(0)
    td = MagicMock()

    def _read(series_ids, retention):
        values = rng.uniform(10, 200, len(idx))
        result = MagicMock()
        result.__len__ = lambda _: len(idx)
        pdf = pd.DataFrame({"valid_time": idx.tz_localize(None), "value": values})
        result.to_pandas.return_value = pdf
        return result

    td.read.side_effect = _read
    return td


def _make_ch_mock(idx):
    """clickhouse-connect mock returning empty Silver tables."""
    ch = MagicMock()
    ch.query_df.return_value = pd.DataFrame()
    return ch


def test_build_features_returns_dataframe(se3):
    idx = pd.date_range("2026-01-01", periods=N_HOURS, freq="h", tz="UTC")
    df = build_features(
        se3,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=N_HOURS),
        td=_make_td_mock(idx),
        ch_client=_make_ch_mock(idx),
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == N_HOURS


def test_build_features_has_valid_time_column(se3):
    idx = pd.date_range("2026-01-01", periods=N_HOURS, freq="h", tz="UTC")
    df = build_features(
        se3,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=N_HOURS),
        td=_make_td_mock(idx),
        ch_client=_make_ch_mock(idx),
    )
    assert "valid_time" in df.columns
    assert df.index.name != "valid_time"  # must be a column, not the index


def test_build_features_has_zone_column(se3):
    idx = pd.date_range("2026-01-01", periods=N_HOURS, freq="h", tz="UTC")
    df = build_features(
        se3,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=N_HOURS),
        td=_make_td_mock(idx),
        ch_client=_make_ch_mock(idx),
    )
    assert "zone" in df.columns
    assert (df["zone"] == se3.entsoe_eic).all()


def test_build_features_zone_column_is_second(se3):
    idx = pd.date_range("2026-01-01", periods=N_HOURS, freq="h", tz="UTC")
    df = build_features(
        se3,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=N_HOURS),
        td=_make_td_mock(idx),
        ch_client=_make_ch_mock(idx),
    )
    assert df.columns[0] == "valid_time"
    assert df.columns[1] == "zone"


def test_build_features_contains_price_lags(se3):
    idx = pd.date_range("2026-01-01", periods=N_HOURS, freq="h", tz="UTC")
    df = build_features(
        se3,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=N_HOURS),
        td=_make_td_mock(idx),
        ch_client=_make_ch_mock(idx),
    )
    for lag in [23, 24, 25, 48, 72, 168, 336]:
        assert f"price_lag{lag}h" in df.columns, f"Missing price_lag{lag}h"


def test_build_features_contains_calendar_cols(se3):
    idx = pd.date_range("2026-01-01", periods=N_HOURS, freq="h", tz="UTC")
    df = build_features(
        se3,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=N_HOURS),
        td=_make_td_mock(idx),
        ch_client=_make_ch_mock(idx),
    )
    for col in ("hour", "weekday", "month", "is_holiday", "hour_sin", "hour_cos"):
        assert col in df.columns, f"Missing calendar column: {col}"
