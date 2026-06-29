"""Unit tests for ml/models/ensemble.py.

Tests train() and predict() on synthetic base model predictions.
The ensemble uses Ridge regression with no external infrastructure —
tests run without any model files on disk (tmp_path isolates writes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ml.models.ensemble as ens_mod

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    """Redirect MODEL_DIR into a temp directory so tests don't touch model/."""
    monkeypatch.setattr(ens_mod, "MODEL_DIR", tmp_path)
    return tmp_path


@pytest.fixture()
def base_preds_and_actuals() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic base model predictions + true prices for meta-learner training."""
    n = 200
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    base_preds = pd.DataFrame(
        {
            "lgbm_q05": rng.uniform(10, 50, n),
            "lgbm_q50": rng.uniform(50, 100, n),
            "lgbm_q95": rng.uniform(100, 200, n),
            "xgb_q05": rng.uniform(10, 50, n),
            "xgb_q50": rng.uniform(50, 100, n),
            "xgb_q95": rng.uniform(100, 200, n),
            "cat_q05": rng.uniform(10, 50, n),
            "cat_q50": rng.uniform(50, 100, n),
            "cat_q95": rng.uniform(100, 200, n),
        },
        index=idx,
    )
    actuals = pd.Series(rng.uniform(30, 150, n), index=idx, name="price")
    return base_preds, actuals


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_train_returns_three_quantile_models(model_dir, base_preds_and_actuals):
    """train() must return exactly the keys q05, q50, q95."""
    base_preds, actuals = base_preds_and_actuals
    models = ens_mod.train(base_preds, actuals)
    assert set(models.keys()) == {"q05", "q50", "q95"}


def test_predict_returns_correct_columns(model_dir, base_preds_and_actuals):
    """predict() must return a DataFrame with columns ens_q05, ens_q50, ens_q95."""
    base_preds, actuals = base_preds_and_actuals
    ens_mod.train(base_preds, actuals)
    preds = ens_mod.predict(base_preds)
    assert list(preds.columns) == ["ens_q05", "ens_q50", "ens_q95"]


def test_predict_length_matches_input(model_dir, base_preds_and_actuals):
    """predict() output must have the same length as the input DataFrame."""
    base_preds, actuals = base_preds_and_actuals
    ens_mod.train(base_preds, actuals)
    preds = ens_mod.predict(base_preds)
    assert len(preds) == len(base_preds)


def test_quantile_ordering_preserved(model_dir, base_preds_and_actuals):
    """predict() must ensure ens_q05 ≤ ens_q50 ≤ ens_q95 for every row."""
    base_preds, actuals = base_preds_and_actuals
    ens_mod.train(base_preds, actuals)
    preds = ens_mod.predict(base_preds)
    assert (
        preds["ens_q05"] <= preds["ens_q50"]
    ).all(), "Quantile ordering violated: ens_q05 > ens_q50 for some rows"
    assert (
        preds["ens_q50"] <= preds["ens_q95"]
    ).all(), "Quantile ordering violated: ens_q50 > ens_q95 for some rows"
