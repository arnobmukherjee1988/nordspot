"""Unit tests for ml/mlflow_setup.py.

All tests use mlflow's in-memory SQLite tracking store via a tmp_path fixture —
no MLflow server required.
"""

from __future__ import annotations

import mlflow
import pytest

from ml.mlflow_setup import EXPERIMENTS, create_experiments, get_tracking_uri

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def local_mlflow(tmp_path, monkeypatch):
    """Point MLflow at a temp SQLite DB for the duration of the test."""
    uri = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    # mlflow client caches the tracking URI; reset between tests
    mlflow.set_tracking_uri("")


# ── EXPERIMENTS constant ──────────────────────────────────────────────────────


def test_experiments_has_five_keys():
    assert len(EXPERIMENTS) == 5


def test_experiments_contains_expected_aliases():
    expected = {"lgbm", "xgboost", "catboost", "lear", "ensemble"}
    assert set(EXPERIMENTS.keys()) == expected


def test_experiment_names_are_prefixed():
    for name in EXPERIMENTS.values():
        assert name.startswith("nordspot-"), f"Bad prefix: {name}"


def test_experiment_names_are_unique():
    names = list(EXPERIMENTS.values())
    assert len(names) == len(set(names)), "Duplicate experiment names detected"


# ── get_tracking_uri ──────────────────────────────────────────────────────────


def test_get_tracking_uri_reads_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://my-server:9999")
    assert get_tracking_uri() == "http://my-server:9999"


def test_get_tracking_uri_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert get_tracking_uri() == "http://localhost:5000"


# ── create_experiments ────────────────────────────────────────────────────────


def test_create_experiments_returns_all_aliases(local_mlflow):
    ids = create_experiments(tracking_uri=local_mlflow)
    assert set(ids.keys()) == set(EXPERIMENTS.keys())


def test_create_experiments_returns_string_ids(local_mlflow):
    ids = create_experiments(tracking_uri=local_mlflow)
    for alias, exp_id in ids.items():
        assert isinstance(exp_id, str), f"{alias}: expected str ID, got {type(exp_id)}"


def test_create_experiments_all_experiments_exist_in_mlflow(local_mlflow):
    create_experiments(tracking_uri=local_mlflow)
    for name in EXPERIMENTS.values():
        exp = mlflow.get_experiment_by_name(name)
        assert (
            exp is not None
        ), f"Experiment '{name}' not found after create_experiments()"


def test_create_experiments_is_idempotent(local_mlflow):
    """Running twice must return the same IDs and not raise."""
    ids_first = create_experiments(tracking_uri=local_mlflow)
    ids_second = create_experiments(tracking_uri=local_mlflow)
    assert ids_first == ids_second


def test_create_experiments_idempotent_experiment_count(local_mlflow):
    """MLflow must not create duplicate experiments on the second call."""
    create_experiments(tracking_uri=local_mlflow)
    create_experiments(tracking_uri=local_mlflow)
    client = mlflow.tracking.MlflowClient(tracking_uri=local_mlflow)
    all_exps = client.search_experiments()
    # Filter to only our nordspot- experiments (exclude Default experiment id=0)
    nordspot_exps = [e for e in all_exps if e.name.startswith("nordspot-")]
    assert len(nordspot_exps) == len(EXPERIMENTS)
