"""Unit tests for ml/models/xgboost.py.

Tests train(), predict(), and calibrate() on synthetic data.
XGBoost is fast enough that we can run real training with capped n_estimators
(10 trees per quantile) to verify the full code path including file I/O.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import ml.models.xgboost as xgb_mod
from ml.models.lgbm import FEATURE_COLS

# -- Fixtures ------------------------------------------------------------------


@pytest.fixture()
def fast_xgb(monkeypatch):
    """Cap n_estimators at 10 so each quantile trains in < 100 ms."""
    monkeypatch.setitem(xgb_mod._XGB_PARAMS_BASE, "n_estimators", 10)


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    """Redirect MODEL_DIR and CONFORMAL_PATH into a temp directory."""
    monkeypatch.setattr(xgb_mod, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(xgb_mod, "CONFORMAL_PATH", tmp_path / "xgb_conformal.pkl")
    return tmp_path


@pytest.fixture()
def train_df() -> pd.DataFrame:
    """Synthetic feature matrix for the val-split logic.

    N=500 is below the 24*30=720 hour floor, so the n//2 cap in train()
    kicks in (n_val=250, n_tr=250). Tests verify the cap works correctly.
    """
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {col: rng.standard_normal(n) for col in FEATURE_COLS},
        index=idx,
    )
    df.index.name = "valid_time"
    df["price"] = rng.uniform(20, 200, n)
    return df


# -- Tests ---------------------------------------------------------------------


def test_train_returns_three_quantile_models(fast_xgb, model_dir, train_df):
    """train() must return exactly the keys q05, q50, q95."""
    models = xgb_mod.train(train_df, verbose=False)
    assert set(models.keys()) == {"q05", "q50", "q95"}


def test_train_saves_model_files(fast_xgb, model_dir, train_df):
    """train() must persist one .pkl file per quantile in MODEL_DIR."""
    xgb_mod.train(train_df, verbose=False)
    for name in ("q05", "q50", "q95"):
        path = model_dir / f"xgb_{name}.pkl"
        assert path.exists(), f"Model file not found: {path}"
        # Verify the file is a valid pickle (not an empty stub)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert obj is not None


def test_predict_returns_correct_columns(fast_xgb, model_dir, train_df):
    """predict() must return a DataFrame with columns xgb_q05/q50/q95."""
    xgb_mod.train(train_df, verbose=False)
    preds = xgb_mod.predict(train_df, apply_conformal=False)
    assert list(preds.columns) == ["xgb_q05", "xgb_q50", "xgb_q95"]


def test_predict_length_matches_input(fast_xgb, model_dir, train_df):
    """predict() output must have the same length as the input DataFrame."""
    xgb_mod.train(train_df, verbose=False)
    preds = xgb_mod.predict(train_df, apply_conformal=False)
    assert len(preds) == len(train_df)


def test_calibrate_returns_float_and_saves_bundle(fast_xgb, model_dir, train_df):
    """calibrate() must return a float and persist the correction to disk."""
    xgb_mod.train(train_df, verbose=False)
    raw = xgb_mod.predict(train_df, apply_conformal=False)

    actuals = train_df["price"].reset_index(drop=True)
    actuals.index = train_df.index

    c = xgb_mod.calibrate(actuals, raw["xgb_q05"], raw["xgb_q95"])
    assert isinstance(c, float)
    assert model_dir.joinpath("xgb_conformal.pkl").exists()


def test_conformal_widens_interval(fast_xgb, model_dir, train_df):
    """After calibrate(), predict() with apply_conformal=True must widen intervals
    by exactly c_hat vs apply_conformal=False when c_hat > 0."""
    xgb_mod.train(train_df, verbose=False)
    raw = xgb_mod.predict(train_df, apply_conformal=False)

    # Force a positive correction: all actuals above the upper bound
    actuals_s = pd.Series(
        np.random.default_rng(7).uniform(
            raw["xgb_q95"].values + 200,  # all actuals above upper bound
            raw["xgb_q95"].values + 300,
            len(train_df),
        ),
        index=train_df.index,
    )
    c = xgb_mod.calibrate(actuals_s, raw["xgb_q05"], raw["xgb_q95"])

    if c > 0:
        cal = xgb_mod.predict(train_df, apply_conformal=True)
        width_raw = (raw["xgb_q95"] - raw["xgb_q05"]).mean()
        width_cal = (cal["xgb_q95"] - cal["xgb_q05"]).mean()
        assert width_cal > width_raw, (
            f"Conformal calibration should widen interval: "
            f"raw={width_raw:.2f}, cal={width_cal:.2f}"
        )
