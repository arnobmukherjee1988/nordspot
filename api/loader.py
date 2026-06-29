"""Model loader - called once at API startup.

Readiness strategy (disk-first):
    1. Check whether the ensemble pickle files exist in MODEL_DIR.
       These are written by ml/train.py regardless of MLflow artifact status.
    2. Optionally enrich store metadata from the MLflow Registry (version,
       trained_at) - best-effort, failures are logged but not fatal.

Why not mlflow.pyfunc.load_model()?
    The API predictor (api/predictor.py) loads models directly from
    MODEL_DIR/*.pkl via ml.models.*.predict() - it never calls store.model.
    store.is_ready is the only gate.  Downloading a PyFunc artefact from
    MLflow is unnecessary and breaks when the MLflow server's artifact root
    is not reachable from the training host (file:///mlflow/... on Docker).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mlflow.tracking import MlflowClient

from ml.mlflow_setup import get_tracking_uri

logger = logging.getLogger("nordspot.api.loader")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
MODEL_NAME = os.getenv("NORDSPOT_MODEL_NAME", "nordspot-ensemble-SE3")

# Sentinel: set on store.model when pickle files are present.
# The predictor never uses this value - only is_ready (not None) matters.
_MODEL_READY_SENTINEL = True


def load_production_models(
    model_name: str = MODEL_NAME,
    tracking_uri: Optional[str] = None,
) -> object:  # returns ModelStore
    """Check disk for model files and enrich metadata from MLflow Registry.

    Returns a ModelStore with is_ready=True when ensemble pickle files exist.
    """
    from api.model_store import ModelStore

    store = ModelStore()

    # -- 1. Disk check (primary gate) -----------------------------------------
    required_files = [
        MODEL_DIR / "ensemble_q05.pkl",
        MODEL_DIR / "ensemble_q50.pkl",
        MODEL_DIR / "ensemble_q95.pkl",
    ]
    missing = [str(f) for f in required_files if not f.exists()]
    if missing:
        logger.warning(
            "Ensemble model files not found: %s - API will serve stub predictions",
            missing,
        )
        return store

    store.model = _MODEL_READY_SENTINEL
    logger.info("Ensemble model files found in %s - API is ready", MODEL_DIR)

    # -- 2. MLflow Registry metadata (best-effort) ----------------------------
    try:
        uri = tracking_uri or get_tracking_uri()
        client = MlflowClient(tracking_uri=uri)
        mv = client.get_model_version_by_alias(model_name, "champion")
        store.model_version = mv.version
        ts_s = mv.creation_timestamp / 1_000
        store.trained_at = datetime.fromtimestamp(ts_s, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        logger.info(
            "MLflow Registry: '%s' champion is v%s  trained_at=%s",
            model_name,
            store.model_version,
            store.trained_at,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MLflow Registry metadata unavailable (%s) - using defaults", exc
        )
        store.model_version = "unknown"

    return store
