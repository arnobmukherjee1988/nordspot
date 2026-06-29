"""Tests for zone-parameterised training (Story 4.8).

Two scenarios:
  1. train(zone='SE1') - the 'zone' MLflow tag on the lgbm run equals 'SE1'
  2. train_all_zones() - train() is called exactly once per Swedish zone (SE1-SE4)
"""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest

from ml.mlflow_setup import EXPERIMENTS
from ml.train import train, train_all_zones

_N = 800
_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
_END = datetime(2024, 6, 1, tzinfo=timezone.utc)


# -- Helpers -------------------------------------------------------------------


def _make_fake_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=_N, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "valid_time": idx,
            "price": rng.uniform(20, 200, _N),
            "zone": "10Y1001A1001A44P",  # SE1 EIC - arbitrary for test purposes
        },
        index=idx,
    )


def _preds(cols: list[str], idx) -> pd.DataFrame:
    """Return a fake predictions DataFrame with the given column names."""
    rng = np.random.default_rng(99)
    return pd.DataFrame({c: rng.uniform(10, 200, _N) for c in cols}, index=idx)


def _run_train_for_zone(fake_df: pd.DataFrame, zone: str) -> None:
    """Call train() with all external dependencies patched, for a given zone."""
    idx = fake_df.index
    _patches = [
        # -- Core infrastructure -----------------------------------
        patch("ml.train.init_schema", return_value=MagicMock()),
        patch("ml.train.build_features", return_value=fake_df),
        patch("ml.train._write_forecasts_to_timedb"),
        patch("ml.train._s3_upload_models"),
        patch("ml.train._log_feature_importance"),
        patch("ml.train.log_shap_artifacts"),
        patch("ml.train.register_and_promote"),
        patch("mlflow.sklearn.log_model"),
        patch("ml.train._append_log"),  # prevent writes to model/MODEL_LOG.md
        # -- LGBM -------------------------------------------------
        patch(
            "ml.models.lgbm.train",
            return_value={
                q: MagicMock(best_iteration_=100, n_estimators=3000)
                for q in ("q05", "q50", "q95")
            },
        ),
        patch(
            "ml.models.lgbm.predict",
            return_value=_preds(["lgbm_q05", "lgbm_q50", "lgbm_q95"], idx),
        ),
        patch("ml.models.lgbm.calibrate", return_value=0.5),
        patch(
            "ml.models.lgbm.feature_importance",
            return_value=pd.DataFrame({"mean": [1.0]}, index=["price_lag24h"]),
        ),
        patch("mlflow.lightgbm.log_model"),
        # -- LEAR -------------------------------------------------
        patch("ml.models.lear.train"),
        patch(
            "ml.models.lear.predict",
            return_value=_preds(["lear_q05", "lear_q50", "lear_q95"], idx),
        ),
        # -- XGBoost ----------------------------------------------
        patch(
            "ml.models.xgboost.train",
            return_value={q: MagicMock() for q in ("q05", "q50", "q95")},
        ),
        patch(
            "ml.models.xgboost.predict",
            return_value=_preds(["xgb_q05", "xgb_q50", "xgb_q95"], idx),
        ),
        patch("ml.models.xgboost.calibrate", return_value=0.3),
        patch("mlflow.xgboost.log_model"),
        # -- CatBoost ---------------------------------------------
        patch(
            "ml.models.catboost.train",
            return_value={q: MagicMock() for q in ("q05", "q50", "q95")},
        ),
        patch(
            "ml.models.catboost.predict",
            return_value=_preds(["cat_q05", "cat_q50", "cat_q95"], idx),
        ),
        patch("ml.models.catboost.calibrate", return_value=0.2),
        patch("mlflow.catboost.log_model"),
        # -- Ensemble ---------------------------------------------
        patch(
            "ml.models.ensemble.train",
            return_value={
                "q05": MagicMock(coef_=[0.3, 0.3, 0.4], intercept_=0.0),
                "q50": MagicMock(coef_=[0.4, 0.3, 0.3], intercept_=0.0),
                "q95": MagicMock(coef_=[0.3, 0.4, 0.3], intercept_=0.0),
            },
        ),
        patch(
            "ml.models.ensemble.predict",
            return_value=_preds(["ens_q05", "ens_q50", "ens_q95"], idx),
        ),
    ]
    with ExitStack() as stack:
        for p in _patches:
            stack.enter_context(p)
        train(start=_START, end=_END, zone=zone, note="zone test")


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture()
def local_mlflow(tmp_path, monkeypatch):
    """Temp SQLite MLflow backend scoped to one test."""
    uri = f"sqlite:///{tmp_path / 'mlflow_zone.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri("")


# -- Tests ---------------------------------------------------------------------


def test_zone_tag_propagates_to_lgbm_run(local_mlflow):
    """train(zone='SE1') must tag the nordspot-lgbm MLflow run with zone='SE1'."""
    fake_df = _make_fake_df()
    _run_train_for_zone(fake_df, zone="SE1")

    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["lgbm"])
    assert exp is not None, f"Experiment '{EXPERIMENTS['lgbm']}' was not created"
    tags = client.search_runs(experiment_ids=[exp.experiment_id])[0].data.tags
    assert (
        tags.get("zone") == "SE1"
    ), f"Expected zone tag 'SE1', got '{tags.get('zone')}'"


def test_train_all_zones_trains_each_zone():
    """train_all_zones() must call train() exactly once per Swedish zone (SE1-SE4)."""
    with patch("ml.train.train") as mock_train:
        train_all_zones(start=_START, end=_END, note="all zones test")

    assert (
        mock_train.call_count == 4
    ), f"Expected 4 train() calls (one per zone), got {mock_train.call_count}"
    zones_trained = {c.kwargs["zone"] for c in mock_train.call_args_list}
    assert zones_trained == {
        "SE1",
        "SE2",
        "SE3",
        "SE4",
    }, f"Expected all four zones to be trained, got: {zones_trained}"
