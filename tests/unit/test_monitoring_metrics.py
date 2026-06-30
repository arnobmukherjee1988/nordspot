"""Unit tests for monitoring/metrics.py.

All tests use synthetic price series so no DB connection is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring.metrics import (
    compute_rolling_metrics,
    coverage_rate,
    mae,
    naive_mae,
    pinball_loss,
    relative_mae,
)


def _make_series(values, start="2026-01-01", freq="h") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


# -- mae -------------------------------------------------------------------


def test_mae_perfect_forecast():
    actuals = _make_series([10.0, 20.0, 30.0])
    forecasts = _make_series([10.0, 20.0, 30.0])
    assert mae(actuals, forecasts) == pytest.approx(0.0)


def test_mae_known_value():
    actuals = _make_series([10.0, 20.0, 30.0])
    forecasts = _make_series([12.0, 18.0, 33.0])
    # errors: 2, 2, 3 -> mean = 2.333...
    assert mae(actuals, forecasts) == pytest.approx(7.0 / 3, rel=1e-6)


def test_mae_nan_dropped():
    actuals = _make_series([10.0, float("nan"), 30.0])
    forecasts = _make_series([12.0, 18.0, 33.0])
    # only hours 0 and 2 align on non-NaN actuals
    assert mae(actuals, forecasts) == pytest.approx((2.0 + 3.0) / 2, rel=1e-6)


def test_mae_too_few_rows_returns_nan():
    actuals = _make_series([10.0])
    forecasts = _make_series([12.0])
    assert np.isnan(mae(actuals, forecasts))


# -- naive_mae -------------------------------------------------------------


def test_naive_mae_constant_series():
    # constant prices -> naive MAE = 0
    actuals = _make_series([50.0] * 48)
    assert naive_mae(actuals) == pytest.approx(0.0)


def test_naive_mae_step_change():
    # first 24 hours: 50, next 24 hours: 70
    values = [50.0] * 24 + [70.0] * 24
    actuals = _make_series(values)
    # naive for hours 24-47: price at hours 0-23 = 50
    # actual for hours 24-47: 70 -> abs error = 20 each
    # hours 0-23 have no lag -> dropped
    assert naive_mae(actuals) == pytest.approx(20.0)


# -- relative_mae ----------------------------------------------------------


def test_rmae_model_beats_naive():
    values = [50.0] * 24 + [70.0] * 24
    actuals = _make_series(values)
    # model is perfect
    forecasts = actuals.copy()
    rmae = relative_mae(actuals, forecasts)
    assert rmae == pytest.approx(0.0)


def test_rmae_model_same_as_naive():
    values = [50.0] * 24 + [70.0] * 24
    actuals = _make_series(values)
    # model forecast = naive (lag 24h)
    naive = actuals.shift(24)
    rmae = relative_mae(actuals, naive)
    assert rmae == pytest.approx(1.0, rel=1e-4)


# -- pinball_loss ----------------------------------------------------------


def test_pinball_q50_equals_half_mae():
    actuals = _make_series([10.0, 20.0, 30.0])
    forecasts = _make_series([12.0, 18.0, 33.0])
    pb = pinball_loss(actuals, forecasts, q=0.50)
    m = mae(actuals, forecasts)
    assert pb == pytest.approx(m / 2, rel=1e-6)


def test_pinball_asymmetry():
    # For q=0.90: under-prediction is penalised 9x more than over-prediction
    actuals = _make_series([10.0, 10.0])
    # hour 0: forecast=8 (under by 2) -> 0.90 * 2 = 1.8
    # hour 1: forecast=12 (over by 2) -> 0.10 * 2 = 0.2
    forecasts = _make_series([8.0, 12.0])
    pb = pinball_loss(actuals, forecasts, q=0.90)
    assert pb == pytest.approx((1.8 + 0.2) / 2, rel=1e-6)


def test_pinball_nan_returns_nan():
    actuals = _make_series([float("nan")])
    forecasts = _make_series([10.0])
    assert np.isnan(pinball_loss(actuals, forecasts, q=0.50))


# -- coverage_rate ---------------------------------------------------------


def test_coverage_all_inside():
    actuals = _make_series([50.0, 60.0, 70.0])
    q05 = _make_series([40.0, 50.0, 60.0])
    q95 = _make_series([60.0, 70.0, 80.0])
    assert coverage_rate(actuals, q05, q95) == pytest.approx(1.0)


def test_coverage_none_inside():
    actuals = _make_series([100.0, 200.0, 300.0])
    q05 = _make_series([40.0, 50.0, 60.0])
    q95 = _make_series([60.0, 70.0, 80.0])
    assert coverage_rate(actuals, q05, q95) == pytest.approx(0.0)


def test_coverage_partial():
    actuals = _make_series([55.0, 200.0])
    q05 = _make_series([40.0, 40.0])
    q95 = _make_series([60.0, 60.0])
    # hour 0 inside, hour 1 outside
    assert coverage_rate(actuals, q05, q95) == pytest.approx(0.5)


# -- compute_rolling_metrics -----------------------------------------------


def test_compute_rolling_metrics_shape():
    n = 24 * 10  # 10 days
    rng = np.random.default_rng(0)
    actuals = _make_series(rng.uniform(30, 80, n))
    q50 = _make_series(actuals.values + rng.normal(0, 5, n))
    q05 = _make_series(q50.values - 10)
    q95 = _make_series(q50.values + 10)

    result = compute_rolling_metrics(actuals, q05, q50, q95, window_days=7)

    expected_keys = {
        "mae_eur",
        "naive_mae_eur",
        "rmae",
        "pinball_q50",
        "pinball_q05",
        "pinball_q95",
        "coverage_rate",
        "n_hours",
    }
    assert set(result.keys()) == expected_keys


def test_compute_rolling_metrics_window_respected():
    n = 24 * 14  # 14 days
    rng = np.random.default_rng(1)
    actuals = _make_series(rng.uniform(30, 80, n))
    q50 = _make_series(actuals.values + rng.normal(0, 5, n))
    q05 = _make_series(q50.values - 10)
    q95 = _make_series(q50.values + 10)

    result_7 = compute_rolling_metrics(actuals, q05, q50, q95, window_days=7)
    result_14 = compute_rolling_metrics(actuals, q05, q50, q95, window_days=14)

    # 7-day window should have at most 168 aligned hours
    assert result_7["n_hours"] <= 7 * 24
    # 14-day window uses all available data
    assert result_14["n_hours"] >= result_7["n_hours"]
