"""MLflow Registry loader — called once at API startup.

Queries the MLflow Model Registry for the current Production version of
"nordspot-ensemble", downloads the artefact, and returns a populated
ModelStore. If no Production version is registered (or MLflow is
unreachable), an empty ModelStore is returned so the API starts cleanly
and serves stub predictions until a model is promoted.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from ml.mlflow_setup import get_tracking_uri

logger = logging.getLogger("nordspot.api.loader")

# Zone-specific model names registered by ml/train.py via register_and_promote().
# Override NORDSPOT_MODEL_NAME in the environment to target a different zone.
MODEL_NAME = os.getenv("NORDSPOT_MODEL_NAME", "nordspot-ensemble-SE3")


def load_production_models(
    model_name: str = MODEL_NAME,
    tracking_uri: Optional[str] = None,
) -> object:  # returns ModelStore — imported lazily to avoid circular dep
    """Load the Production ensemble model from the MLflow Registry.

    Args:
        model_name:   Registered model name.  Defaults to MODEL_NAME.
        tracking_uri: Override the MLflow tracking URI.  Defaults to the
                      value returned by ml.mlflow_setup.get_tracking_uri().

    Returns:
        A ModelStore instance.  ``is_ready`` is True only when a Production
        model was found and downloaded successfully.
    """
    from api.model_store import ModelStore

    uri = tracking_uri or get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    store = ModelStore()

    # ── 1. Fetch the Production version descriptor ────────────────────────
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MLflow Registry unreachable (%s) — API will serve stub predictions",
            exc,
        )
        return store

    if not prod_versions:
        logger.warning(
            "No Production model registered for '%s' — API will serve stub predictions",
            model_name,
        )
        return store

    mv = prod_versions[0]

    # ── 2. Download the model artefact ────────────────────────────────────
    try:
        model_uri = f"models:/{model_name}/Production"
        store.model = mlflow.pyfunc.load_model(model_uri)
        store.model_version = mv.version
        # creation_timestamp is epoch milliseconds
        ts_s = mv.creation_timestamp / 1_000
        store.trained_at = datetime.fromtimestamp(ts_s, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        logger.info(
            "Loaded '%s' v%s  trained_at=%s",
            model_name,
            store.model_version,
            store.trained_at,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to load artefact for '%s': %s — API will serve stub predictions",
            model_name,
            exc,
        )
        # store remains not_ready (model is still None)

    return store
