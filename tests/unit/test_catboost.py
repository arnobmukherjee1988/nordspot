"""Unit tests for ml/models/catboost.py.

Tests train(), predict(), and calibrate() on synthetic data.
CatBoost is fast enough with iterations=10 to verify the full code path
including Pool construction, .cbm file I/O, and conformal calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ml.models.catboost as cat_mod
from ml.models.lgbm import FEATURE_COLS

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def fast_cb(monkeypatch):
    """Cap iterations at 10 so each quantile trains in < 500 ms."""
    monkeypatch.setitem(cat_mod._CB_PARAMS_BASE, "iterations", 10)


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    """Redirect MODEL_DIR and CONFORMAL_PATH into a temp directory."""
    monkeypatch.setattr(cat_mod, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(cat_mod, "CONFORMAL_PATH", tmp_path / "cat_conformal.pkl")
    return tmp_path


@pytest.fixture()
def train_df() -> pd.DataFrame:
    """Synthetic feature matrix with correct dtypes for categorical columns."""
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {col: rng.standard_normal(n) for col in FEATURE_COLS},
        index=idx,
    )
    df.index.name = "valid_time"

    # Categorical features must be integers — set realistic ranges
    df["hour"] = (idx.hour).astype(int)
    df["weekday"] = (idx.dayofweek).astype(int)
    df["month"] = (idx.month).astype(int)

    df["price"] = rng.uniform(20, 200, n)
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_train_returns_three_quantile_models(fast_cb, model_dir, train_df):
    """train() must return exactly the keys q05, q50, q95."""
    models = cat_mod.train(train_df, verbose=False)
    assert set(models.keys()) == {"q05", "q50", "q95"}


def test_train_saves_cbm_files(fast_cb, model_dir, train_df):
    """train() must save one .cbm file per quantile in MODEL_DIR."""
    cat_mod.train(train_df, verbose=False)
    for name in ("q05", "q50", "q95"):
        path = model_dir / f"cat_{name}.cbm"
        assert path.exists(), f"Model file not found: {path}"
        assert path.stat().st_size > 0, f"Model file is empty: {path}"


def test_predict_returns_correct_columns(fast_cb, model_dir, train_df):
    """predict() must return a DataFrame with columns cat_q05/q50/q95."""
    cat_mod.train(train_df, verbose=False)
    preds = cat_mod.predict(train_df, apply_conformal=False)
    assert list(preds.columns) == ["cat_q05", "cat_q50", "cat_q95"]


def test_predict_length_matches_input(fast_cb, model_dir, train_df):
    """predict() output must have the same length as the input DataFrame."""
    cat_mod.train(train_df, verbose=False)
    preds = cat_mod.predict(train_df, apply_conformal=False)
    assert len(preds) == len(train_df)


def test_calibrate_returns_float_and_saves_bundle(fast_cb, model_dir, train_df):
    """calibrate() must return a float and persist the bundle to disk."""
    cat_mod.train(train_df, verbose=False)
    raw = cat_mod.predict(train_df, apply_conformal=False)

    actuals = train_df["price"].copy()
    c = cat_mod.calibrate(actuals, raw["cat_q05"], raw["cat_q95"])

    assert isinstance(c, float)
    assert (model_dir / "cat_conformal.pkl").exists()


def test_conformal_widens_interval(fast_cb, model_dir, train_df):
    """After calibrate() with positive ĉ, predict() must widen the interval."""
    cat_mod.train(train_df, verbose=False)
    raw = cat_mod.predict(train_df, apply_conformal=False)

    # Force a positive correction: all actuals above the upper bound
    actuals_s = pd.Series(
        np.random.default_rng(7).uniform(
            raw["cat_q95"].values + 200,
            raw["cat_q95"].values + 300,
            len(train_df),
        ),
        index=train_df.index,
    )
    c = cat_mod.calibrate(actuals_s, raw["cat_q05"], raw["cat_q95"])

    if c > 0:
        cal = cat_mod.predict(train_df, apply_conformal=True)
        width_raw = (raw["cat_q95"] - raw["cat_q05"]).mean()
        width_cal = (cal["cat_q95"] - cal["cat_q05"]).mean()
        assert width_cal > width_raw, (
            f"Conformal calibration should widen interval: "
            f"raw={width_raw:.2f}, cal={width_cal:.2f}"
        )
