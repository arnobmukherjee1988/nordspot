"""Unit tests for api/loader.py - disk-first model readiness.

Readiness is gated on ensemble pickle files existing in MODEL_DIR.
MLflow Registry is used best-effort for metadata only.

All external I/O (Path.exists, MlflowClient, get_tracking_uri) is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlflow

from api.loader import load_production_models
from api.model_store import ModelStore

# -- Shared constants ----------------------------------------------------------

_EPOCH_MS = 1_700_000_000_000  # 2023-11-14T22:13:20 UTC
_ISO = "2023-11-14T22:13:20Z"
_VERSION = "3"

# -- Helpers -------------------------------------------------------------------


def _mv(version: str = _VERSION, ts_ms: int = _EPOCH_MS) -> MagicMock:
    """Build a minimal mock ModelVersion descriptor with champion alias."""
    mv = MagicMock()
    mv.version = version
    mv.creation_timestamp = ts_ms
    return mv


def _client_with_champion(version: str = _VERSION, ts_ms: int = _EPOCH_MS) -> MagicMock:
    c = MagicMock()
    c.get_model_version_by_alias.return_value = _mv(version=version, ts_ms=ts_ms)
    return c


def _client_no_champion() -> MagicMock:
    c = MagicMock()
    c.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException(
        "No alias"
    )
    return c


def _client_unreachable() -> MagicMock:
    c = MagicMock()
    c.get_model_version_by_alias.side_effect = Exception("connection refused")
    return c


# Files that must exist for the API to be ready
_REQUIRED = [
    "model/ensemble_q05.pkl",
    "model/ensemble_q50.pkl",
    "model/ensemble_q95.pkl",
]


# -- Disk gate (primary readiness) ---------------------------------------------


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.Path.exists", return_value=False)
def test_missing_model_files_returns_not_ready(_exists, mock_client_cls, _gtu):
    mock_client_cls.return_value = _client_with_champion()
    store = load_production_models()
    assert not store.is_ready


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.Path.exists", return_value=True)
def test_present_model_files_returns_ready(_exists, mock_client_cls, _gtu):
    mock_client_cls.return_value = _client_with_champion()
    store = load_production_models()
    assert store.is_ready


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.Path.exists", return_value=False)
def test_missing_files_keeps_default_model_version(_exists, mock_client_cls, _gtu):
    mock_client_cls.return_value = _client_with_champion()
    store = load_production_models()
    assert store.model_version == "not_loaded"


# -- MLflow metadata enrichment (best-effort) ----------------------------------


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.Path.exists", return_value=True)
def test_mlflow_champion_sets_version(_exists, mock_client_cls, _gtu):
    mock_client_cls.return_value = _client_with_champion(version="7")
    store = load_production_models()
    assert store.model_version == "7"


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.Path.exists", return_value=True)
def test_mlflow_champion_sets_trained_at(_exists, mock_client_cls, _gtu):
    mock_client_cls.return_value = _client_with_champion(ts_ms=_EPOCH_MS)
    store = load_production_models()
    assert store.trained_at == _ISO


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.Path.exists", return_value=True)
def test_mlflow_no_champion_still_ready(_exists, mock_client_cls, _gtu):
    """Files present but no champion alias -> ready with unknown version."""
    mock_client_cls.return_value = _client_no_champion()
    store = load_production_models()
    assert store.is_ready
    assert store.model_version == "unknown"


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.Path.exists", return_value=True)
def test_mlflow_unreachable_still_ready(_exists, mock_client_cls, _gtu):
    """Files present but MLflow unreachable -> ready with unknown version."""
    mock_client_cls.return_value = _client_unreachable()
    store = load_production_models()
    assert store.is_ready
    assert store.model_version == "unknown"


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.Path.exists", return_value=True)
def test_tracking_uri_override_is_respected(_exists, mock_client_cls, _gtu):
    mock_client_cls.return_value = _client_with_champion()
    custom_uri = "http://custom-mlflow:9999"
    load_production_models(tracking_uri=custom_uri)
    mock_client_cls.assert_called_once_with(tracking_uri=custom_uri)


# -- ModelStore property -------------------------------------------------------


def test_model_store_is_ready_false_by_default():
    assert not ModelStore().is_ready


def test_model_store_is_ready_true_when_model_set():
    assert ModelStore(model=MagicMock(), model_version="1").is_ready
