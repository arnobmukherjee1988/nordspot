"""Tests for MLflow integration in ml/train.py.

Verifies that train() creates an MLflow run in the correct experiment,
logs the expected metrics and hyperparameter params, and finishes cleanly.

All external dependencies (TimeDB, ClickHouse, LightGBM model files,
LEAR model files, S3) are patched so no infrastructure is required.
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
from ml.train import train

# ── Synthetic data helpers ────────────────────────────────────────────────────

_N = 800  # rows — enough for _compute_metrics (needs > 720)

_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
_END = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _fake_df() -> pd.DataFrame:
    """Feature DataFrame large enough that labelled < 720 check passes."""
    idx = pd.date_range("2024-01-01", periods=_N, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "valid_time": idx,
            "price": rng.uniform(20, 200, _N),
            "zone": "10Y1001A1001A46L",
        },
        index=idx,
    )


def _fake_lgbm_preds(df: pd.DataFrame, **_kw: object) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = len(df)
    return pd.DataFrame(
        {
            "lgbm_q05": rng.uniform(10, 60, n),
            "lgbm_q50": rng.uniform(30, 120, n),
            "lgbm_q95": rng.uniform(100, 250, n),
        },
        index=df.index,
    )


def _fake_lear_preds(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = len(df)
    return pd.DataFrame(
        {
            "lear_q05": rng.uniform(10, 60, n),
            "lear_q50": rng.uniform(30, 120, n),
            "lear_q95": rng.uniform(100, 250, n),
        },
        index=df.index,
    )


def _fake_xgb_preds(df: pd.DataFrame, **_kw: object) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = len(df)
    return pd.DataFrame(
        {
            "xgb_q05": rng.uniform(10, 60, n),
            "xgb_q50": rng.uniform(30, 120, n),
            "xgb_q95": rng.uniform(100, 250, n),
        },
        index=df.index,
    )


def _fake_cat_preds(df: pd.DataFrame, **_kw: object) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = len(df)
    return pd.DataFrame(
        {
            "cat_q05": rng.uniform(10, 60, n),
            "cat_q50": rng.uniform(30, 120, n),
            "cat_q95": rng.uniform(100, 250, n),
        },
        index=df.index,
    )


def _fake_ens_preds(base_preds: pd.DataFrame, **_kw: object) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = len(base_preds)
    return pd.DataFrame(
        {
            "ens_q05": rng.uniform(10, 60, n),
            "ens_q50": rng.uniform(30, 120, n),
            "ens_q95": rng.uniform(100, 250, n),
        },
        index=base_preds.index,
    )


def _fake_fi() -> pd.DataFrame:
    """Minimal feature-importance DataFrame for lgbm.feature_importance mock."""
    return pd.DataFrame({"mean": [1.0, 0.5]}, index=["price_lag24h", "hour"])


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def local_mlflow(tmp_path, monkeypatch):
    """Temp SQLite MLflow backend scoped to one test."""
    uri = f"sqlite:///{tmp_path / 'mlflow_train.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri("")


# ── Helper: run train() with all external deps patched ────────────────────────


def _run_train(note: str = "test run") -> None:
    """Call train() with infrastructure patched out.

    The only real code paths exercised are:
      • mlflow.start_run / log_params / log_metrics / set_tags
      • _compute_metrics (pure maths — no side effects)
      • _append_log (writes MODEL_LOG.md, harmless in tests)
      • _print_metrics_table (stdout only)
    """
    fake_df = _fake_df()

    # ExitStack avoids Python's "too many statically nested blocks" error
    # that triggers when a parenthesised `with` block exceeds ~20 context managers.
    _patches = [
        # ── Core infrastructure ───────────────────────────────────
        patch("ml.train.init_schema", return_value=MagicMock()),
        patch("ml.train.build_features", return_value=fake_df),
        patch("ml.train._write_forecasts_to_timedb"),
        patch("ml.train._s3_upload_models"),
        patch("ml.train._log_feature_importance"),  # skip matplotlib in CI
        patch("ml.train.log_shap_artifacts"),  # skip SHAP compute in CI
        # ── LGBM ─────────────────────────────────────────────────
        patch(
            "ml.models.lgbm.train",
            return_value={
                "q05": MagicMock(best_iteration_=200, n_estimators=3000),
                "q50": MagicMock(best_iteration_=180, n_estimators=3000),
                "q95": MagicMock(best_iteration_=220, n_estimators=3000),
            },
        ),
        patch("ml.models.lgbm.predict", side_effect=_fake_lgbm_preds),
        patch("ml.models.lgbm.calibrate", return_value=0.5),
        patch("ml.models.lgbm.feature_importance", return_value=_fake_fi()),
        patch("mlflow.lightgbm.log_model"),
        # ── LEAR ─────────────────────────────────────────────────
        patch("ml.models.lear.train"),
        patch("ml.models.lear.predict", side_effect=_fake_lear_preds),
        # ── XGBoost (Story 4.3) ───────────────────────────────────
        patch(
            "ml.models.xgboost.train",
            return_value={
                "q05": MagicMock(),
                "q50": MagicMock(),
                "q95": MagicMock(),
            },
        ),
        patch("ml.models.xgboost.predict", side_effect=_fake_xgb_preds),
        patch("ml.models.xgboost.calibrate", return_value=0.3),
        patch("mlflow.xgboost.log_model"),
        # ── CatBoost (Story 4.4) ──────────────────────────────────
        patch(
            "ml.models.catboost.train",
            return_value={
                "q05": MagicMock(),
                "q50": MagicMock(),
                "q95": MagicMock(),
            },
        ),
        patch("ml.models.catboost.predict", side_effect=_fake_cat_preds),
        patch("ml.models.catboost.calibrate", return_value=0.2),
        patch("mlflow.catboost.log_model"),
        # ── Ensemble (Story 4.6) ──────────────────────────────────────────────
        patch(
            "ml.models.ensemble.train",
            return_value={
                "q05": MagicMock(coef_=[0.3, 0.3, 0.4], intercept_=0.0),
                "q50": MagicMock(coef_=[0.4, 0.3, 0.3], intercept_=0.0),
                "q95": MagicMock(coef_=[0.3, 0.4, 0.3], intercept_=0.0),
            },
        ),
        patch("ml.models.ensemble.predict", side_effect=_fake_ens_preds),
    ]

    with ExitStack() as stack:
        for p in _patches:
            stack.enter_context(p)
        train(start=_START, end=_END, note=note)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_train_creates_mlflow_run(local_mlflow):
    """train() must create exactly one MLflow run."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["lgbm"])
    assert exp is not None, f"Experiment '{EXPERIMENTS['lgbm']}' was not created"
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1, f"Expected 1 run, found {len(runs)}"


def test_train_run_in_correct_experiment(local_mlflow):
    """The run must land in the nordspot-lgbm experiment, not Default."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["lgbm"])
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert runs, f"No runs found in '{EXPERIMENTS['lgbm']}'"


def test_train_run_status_is_finished(local_mlflow):
    """Run must complete cleanly — status FINISHED, not FAILED or RUNNING."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["lgbm"])
    run = client.search_runs(experiment_ids=[exp.experiment_id])[0]
    assert run.info.status == "FINISHED", f"Unexpected run status: {run.info.status}"


def test_train_logs_lgbm_metrics(local_mlflow):
    """All LightGBM test-window metrics must be present and numeric."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["lgbm"])
    metrics = client.search_runs(experiment_ids=[exp.experiment_id])[0].data.metrics
    for key in (
        "lgbm_mae",
        "lgbm_rmse",
        "lgbm_mape",
        "lgbm_coverage",
        "lgbm_spike_mae",
        "lgbm_night_mae",
        "lgbm_peak_mae",
    ):
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(
            metrics[key], float
        ), f"{key} should be float, got {type(metrics[key])}"


def test_train_logs_lear_metrics(local_mlflow):
    """LEAR test-window metrics must also be logged."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["lgbm"])
    metrics = client.search_runs(experiment_ids=[exp.experiment_id])[0].data.metrics
    for key in ("lear_mae", "lear_rmse", "lear_coverage", "lear_spike_mae"):
        assert key in metrics, f"Missing metric: {key}"


def test_train_logs_lgbm_hyperparams(local_mlflow):
    """Key LightGBM hyperparameters must be logged as MLflow params."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["lgbm"])
    params = client.search_runs(experiment_ids=[exp.experiment_id])[0].data.params
    for key in (
        "num_leaves",
        "min_child_samples",
        "learning_rate",
        "lgbm_val_frac",
        "lgbm_early_stop_rounds",
    ):
        assert key in params, f"Missing hyperparameter: {key}"


def test_train_run_has_note_tag(local_mlflow):
    """The 'note' passed to train() must appear as an MLflow tag."""
    _run_train(note="epic 4 story 4.2 test")
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["lgbm"])
    tags = client.search_runs(experiment_ids=[exp.experiment_id])[0].data.tags
    assert tags.get("note") == "epic 4 story 4.2 test"


def test_train_run_has_zone_tag(local_mlflow):
    """The zone tag must be set — used for filtering runs in the MLflow UI."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["lgbm"])
    tags = client.search_runs(experiment_ids=[exp.experiment_id])[0].data.tags
    assert "zone" in tags
    assert tags["zone"] == "SE3"


# ── XGBoost MLflow tests (Story 4.3) ─────────────────────────────────────────


def test_xgboost_run_created(local_mlflow):
    """train() must also create a run in the nordspot-xgboost experiment."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["xgboost"])
    assert exp is not None, f"Experiment '{EXPERIMENTS['xgboost']}' was not created"
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1, f"Expected 1 XGBoost run, found {len(runs)}"


def test_xgboost_run_status_finished(local_mlflow):
    """XGBoost run must complete cleanly — status FINISHED."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["xgboost"])
    run = client.search_runs(experiment_ids=[exp.experiment_id])[0]
    assert run.info.status == "FINISHED", f"XGBoost run status: {run.info.status}"


def test_xgboost_logs_metrics(local_mlflow):
    """Key XGBoost test-window metrics must be present and numeric."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["xgboost"])
    metrics = client.search_runs(experiment_ids=[exp.experiment_id])[0].data.metrics
    for key in ("xgb_mae", "xgb_rmse", "xgb_coverage", "xgb_spike_mae"):
        assert key in metrics, f"Missing XGBoost metric: {key}"
        assert isinstance(metrics[key], float), f"{key} should be float"


# ── CatBoost MLflow tests (Story 4.4) ────────────────────────────────────────


def test_catboost_run_created(local_mlflow):
    """train() must also create a run in the nordspot-catboost experiment."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["catboost"])
    assert exp is not None, f"Experiment '{EXPERIMENTS['catboost']}' was not created"
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1, f"Expected 1 CatBoost run, found {len(runs)}"


def test_catboost_run_status_finished(local_mlflow):
    """CatBoost run must complete cleanly — status FINISHED."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["catboost"])
    run = client.search_runs(experiment_ids=[exp.experiment_id])[0]
    assert run.info.status == "FINISHED", f"CatBoost run status: {run.info.status}"


def test_catboost_logs_metrics(local_mlflow):
    """Key CatBoost test-window metrics must be present and numeric."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["catboost"])
    metrics = client.search_runs(experiment_ids=[exp.experiment_id])[0].data.metrics
    for key in ("cat_mae", "cat_rmse", "cat_coverage", "cat_spike_mae"):
        assert key in metrics, f"Missing CatBoost metric: {key}"
        assert isinstance(metrics[key], float), f"{key} should be float"


# ── Ensemble MLflow tests (Story 4.6) ────────────────────────────────────────


def test_ensemble_run_created(local_mlflow):
    """train() must also create a run in the nordspot-ensemble experiment."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["ensemble"])
    assert exp is not None, f"Experiment '{EXPERIMENTS['ensemble']}' was not created"
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1, f"Expected 1 ensemble run, found {len(runs)}"


def test_ensemble_run_status_finished(local_mlflow):
    """Ensemble run must complete cleanly — status FINISHED."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["ensemble"])
    run = client.search_runs(experiment_ids=[exp.experiment_id])[0]
    assert run.info.status == "FINISHED", f"Ensemble run status: {run.info.status}"


def test_ensemble_logs_metrics(local_mlflow):
    """Key ensemble test-window metrics must be present and numeric."""
    _run_train()
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    exp = client.get_experiment_by_name(EXPERIMENTS["ensemble"])
    metrics = client.search_runs(experiment_ids=[exp.experiment_id])[0].data.metrics
    for key in ("ens_mae", "ens_rmse", "ens_coverage", "ens_spike_mae"):
        assert key in metrics, f"Missing ensemble metric: {key}"
        assert isinstance(metrics[key], float), f"{key} should be float"
