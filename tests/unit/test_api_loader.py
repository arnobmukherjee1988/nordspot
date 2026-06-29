"""Unit tests for api/loader.py — MLflow Registry model loading.

All tests run without a real MLflow server.  MlflowClient, mlflow.pyfunc,
and get_tracking_uri are fully mocked via unittest.mock.patch so no network
calls or local tracking directory accesses are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from api.loader import MODEL_NAME, load_production_models
from api.model_store import ModelStore

# ── Shared constants ──────────────────────────────────────────────────────────

_EPOCH_MS = 1_700_000_000_000  # 2023-11-14T22:13:20 UTC
_ISO = "2023-11-14T22:13:20Z"
_VERSION = "3"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _mv(version: str = _VERSION, ts_ms: int = _EPOCH_MS) -> MagicMock:
    """Build a minimal mock ModelVersion descriptor."""
    mv = MagicMock()
    mv.version = version
    mv.creation_timestamp = ts_ms
    return mv


def _client(prod_versions=None, raise_on_get: bool = False) -> MagicMock:
    """Build a mock MlflowClient."""
    c = MagicMock()
    if raise_on_get:
        c.get_latest_versions.side_effect = Exception("connection refused")
    else:
        c.get_latest_versions.return_value = (
            prod_versions if prod_versions is not None else []
        )
    return c


# ── Error / empty paths ───────────────────────────────────────────────────────


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.mlflow")
def test_no_production_version_returns_not_ready(mock_mlflow, mock_client_cls, _gtu):
    mock_client_cls.return_value = _client(prod_versions=[])

    store = load_production_models()

    assert not store.is_ready


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.mlflow")
def test_no_production_version_keeps_default_model_version(
    mock_mlflow, mock_client_cls, _gtu
):
    mock_client_cls.return_value = _client(prod_versions=[])

    store = load_production_models()

    assert store.model_version == "not_loaded"


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.mlflow")
def test_mlflow_unreachable_returns_not_ready(mock_mlflow, mock_client_cls, _gtu):
    mock_client_cls.return_value = _client(raise_on_get=True)

    store = load_production_models()

    assert not store.is_ready


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.mlflow")
def test_model_artefact_load_failure_returns_not_ready(
    mock_mlflow, mock_client_cls, _gtu
):
    mock_client_cls.return_value = _client(prod_versions=[_mv()])
    mock_mlflow.pyfunc.load_model.side_effect = Exception("artefact missing")

    store = load_production_models()

    assert not store.is_ready


# ── Happy path ────────────────────────────────────────────────────────────────


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.mlflow")
def test_production_exists_returns_ready(mock_mlflow, mock_client_cls, _gtu):
    mock_client_cls.return_value = _client(prod_versions=[_mv()])
    mock_mlflow.pyfunc.load_model.return_value = MagicMock()

    store = load_production_models()

    assert store.is_ready


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.mlflow")
def test_production_sets_model_version(mock_mlflow, mock_client_cls, _gtu):
    mock_client_cls.return_value = _client(prod_versions=[_mv(version="7")])
    mock_mlflow.pyfunc.load_model.return_value = MagicMock()

    store = load_production_models(model_name=MODEL_NAME)

    assert store.model_version == "7"


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.mlflow")
def test_production_sets_trained_at_iso(mock_mlflow, mock_client_cls, _gtu):
    mock_client_cls.return_value = _client(prod_versions=[_mv(ts_ms=_EPOCH_MS)])
    mock_mlflow.pyfunc.load_model.return_value = MagicMock()

    store = load_production_models()

    assert store.trained_at == _ISO


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.mlflow")
def test_production_stores_model_object(mock_mlflow, mock_client_cls, _gtu):
    mock_model = MagicMock()
    mock_client_cls.return_value = _client(prod_versions=[_mv()])
    mock_mlflow.pyfunc.load_model.return_value = mock_model

    store = load_production_models()

    assert store.model is mock_model


@patch("api.loader.get_tracking_uri", return_value="http://mock:5000")
@patch("api.loader.MlflowClient")
@patch("api.loader.mlflow")
def test_tracking_uri_override_is_respected(mock_mlflow, mock_client_cls, _gtu):
    mock_client_cls.return_value = _client(prod_versions=[])
    custom_uri = "http://custom-mlflow:9999"

    load_production_models(tracking_uri=custom_uri)

    mock_mlflow.set_tracking_uri.assert_called_once_with(custom_uri)
    mock_client_cls.assert_called_once_with(tracking_uri=custom_uri)


# ── ModelStore property ───────────────────────────────────────────────────────


def test_model_store_is_ready_false_by_default():
    assert not ModelStore().is_ready


def test_model_store_is_ready_true_when_model_set():
    assert ModelStore(model=MagicMock(), model_version="1").is_ready
