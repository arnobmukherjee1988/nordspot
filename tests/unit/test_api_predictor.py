"""Unit tests for api/predictor.py — full inference chain.

All base models and the ensemble meta-learner are patched at their source
modules (ml.models.*) so no pickle files or database connections are needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from api.predictor import run_inference

# ── Synthetic data ─────────────────────────────────────────────────────────────

_N = 24

_FEATURES = pd.DataFrame(
    {
        "valid_time": pd.date_range("2026-07-01", periods=_N, freq="h", tz="UTC"),
        "zone": "10Y1001A1001A46L",
        "price_lag24h": [48.0] * _N,
        "hour": list(range(_N)),
    }
)

_LGBM_PREDS = pd.DataFrame(
    {"lgbm_q05": [30.0] * _N, "lgbm_q50": [50.0] * _N, "lgbm_q95": [70.0] * _N}
)
_XGB_PREDS = pd.DataFrame(
    {"xgb_q05": [28.0] * _N, "xgb_q50": [48.0] * _N, "xgb_q95": [68.0] * _N}
)
_CAT_PREDS = pd.DataFrame(
    {"cat_q05": [32.0] * _N, "cat_q50": [52.0] * _N, "cat_q95": [72.0] * _N}
)
_ENS_PREDS = pd.DataFrame(
    {"ens_q05": [29.0] * _N, "ens_q50": [50.0] * _N, "ens_q95": [71.0] * _N}
)

# All 9 base-model columns that the ensemble expects
_BASE_COLS = [
    "lgbm_q05",
    "lgbm_q50",
    "lgbm_q95",
    "xgb_q05",
    "xgb_q50",
    "xgb_q95",
    "cat_q05",
    "cat_q50",
    "cat_q95",
]

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mocked_pipeline():
    """Patch all base models and the ensemble at their source modules."""
    with (
        patch("ml.models.lgbm.predict", return_value=_LGBM_PREDS) as mock_lgbm,
        patch("ml.models.xgboost.predict", return_value=_XGB_PREDS) as mock_xgb,
        patch("ml.models.catboost.predict", return_value=_CAT_PREDS) as mock_cat,
        patch("ml.models.ensemble.predict", return_value=_ENS_PREDS) as mock_ens,
    ):
        yield mock_lgbm, mock_xgb, mock_cat, mock_ens


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_lgbm_predict_called_once(mocked_pipeline):
    mock_lgbm, *_ = mocked_pipeline
    run_inference(_FEATURES)
    mock_lgbm.assert_called_once()


def test_xgb_predict_called_once(mocked_pipeline):
    _, mock_xgb, *_ = mocked_pipeline
    run_inference(_FEATURES)
    mock_xgb.assert_called_once()


def test_cat_predict_called_once(mocked_pipeline):
    _, _, mock_cat, _ = mocked_pipeline
    run_inference(_FEATURES)
    mock_cat.assert_called_once()


def test_ensemble_predict_called_once(mocked_pipeline):
    *_, mock_ens = mocked_pipeline
    run_inference(_FEATURES)
    mock_ens.assert_called_once()


def test_ensemble_receives_all_nine_base_columns(mocked_pipeline):
    *_, mock_ens = mocked_pipeline
    run_inference(_FEATURES)
    base_df_arg = mock_ens.call_args[0][0]
    assert list(base_df_arg.columns) == _BASE_COLS


def test_output_has_ens_columns(mocked_pipeline):
    result = run_inference(_FEATURES)
    assert list(result.columns) == ["ens_q05", "ens_q50", "ens_q95"]


def test_output_is_ensemble_result(mocked_pipeline):
    result = run_inference(_FEATURES)
    pd.testing.assert_frame_equal(result, _ENS_PREDS)


def test_output_has_24_rows(mocked_pipeline):
    result = run_inference(_FEATURES)
    assert len(result) == _N
