"""MLflow Model Registry: automatic promotion of the NordSpot ensemble model.

After each training run, register_and_promote() compares the new ensemble's MAE
against the currently Production-registered model and promotes automatically:

    new MAE < prod MAE  ->  new version -> Production, old version -> Archived
    new MAE >= prod MAE  ->  new version -> Staging   (tagged "challenger")
    no Production yet   ->  new version -> Production unconditionally

This implements the standard MLOps "shadow promotion" pattern - regressions
cannot reach Production without being detected by the MAE gate.

The MAE is written as a version tag so comparisons are self-contained: no
external database is needed beyond the MLflow Registry itself.

Stages used:
    Production  - current serving model
    Staging     - challenger: better than nothing, but not beating Production
    Archived    - superseded Production versions (kept for rollback audit)
"""

from __future__ import annotations

import time

import mlflow
from mlflow.tracking import MlflowClient

from ml.mlflow_setup import get_tracking_uri

# Artifact path logged inside the ensemble MLflow run - must match ml/train.py
_ENSEMBLE_ARTIFACT_PATH = "ensemble_q50"

_READY_POLL_INTERVAL_S = 1
_READY_POLL_RETRIES = 30


def register_and_promote(
    run_id: str,
    mae: float,
    model_name: str = "nordspot-ensemble",
    artifact_path: str = _ENSEMBLE_ARTIFACT_PATH,
    tracking_uri: str | None = None,
) -> dict:
    """Register a new ensemble version and auto-promote it if MAE improves.

    Decision logic:
        1. Create the registered model if it does not yet exist.
        2. Register a new version from runs:/{run_id}/{artifact_path}.
        3. Wait up to 30 s for the version to reach READY state.
        4. Tag the version with its MAE for future comparisons.
        5. Fetch the current Production version and compare MAEs.
        6. Promote (-> Production) or demote (-> Staging / "challenger").

    Args:
        run_id:        MLflow run ID from the ensemble training block.
        mae:           Test-window MAE for the new model (lower is better).
        model_name:    Registered model name (default "nordspot-ensemble").
        artifact_path: Artifact path logged inside the run.
        tracking_uri:  Override tracking URI (default: env / localhost:5000).

    Returns:
        dict with keys:
            version  (str)  - new version number
            mae      (float) - MAE of the new version
            action   (str)  - "promoted" | "challenger"
            prev_mae (float, optional) - MAE of the prior Production version
            prod_mae (float, optional) - same as prev_mae when challenger
    """
    uri = tracking_uri or get_tracking_uri()
    client = MlflowClient(tracking_uri=uri)

    # -- 1. Ensure the registered model exists (idempotent) --------------------
    try:
        client.create_registered_model(model_name)
        print(f"  [OK] Created registered model '{model_name}'")
    except mlflow.exceptions.MlflowException:
        pass  # Already exists - expected on every run after the first

    # -- 2. Register new version -----------------------------------------------
    source = f"runs:/{run_id}/{artifact_path}"
    mv = client.create_model_version(name=model_name, source=source, run_id=run_id)
    version = mv.version
    print(f"  [OK] Registered v{version}  source={source}")

    # -- 3. Wait for READY -----------------------------------------------------
    for _ in range(_READY_POLL_RETRIES):
        mv = client.get_model_version(model_name, version)
        if mv.status == "READY":
            break
        time.sleep(_READY_POLL_INTERVAL_S)

    # -- 4. Tag new version with MAE -------------------------------------------
    client.set_model_version_tag(model_name, version, "mae", str(round(mae, 4)))
    client.set_model_version_tag(model_name, version, "run_id_short", run_id[:8])

    # -- 5. Fetch current Production (if any) ----------------------------------
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])

    if not prod_versions:
        # No Production yet - promote unconditionally
        client.transition_model_version_stage(model_name, version, "Production")
        client.set_model_version_tag(model_name, version, "registry_action", "promoted")
        print(
            f"  [OK] v{version} -> Production  "
            f"(MAE={mae:.4f}, no prior production model)"
        )
        return {"version": version, "mae": mae, "action": "promoted"}

    prod_mv = prod_versions[0]
    prod_mae_tag = prod_mv.tags.get("mae")

    if prod_mae_tag is None:
        # Production exists but has no MAE tag (registered outside this system).
        # Treat as unknown quality - promote the new version and archive the old.
        client.transition_model_version_stage(model_name, prod_mv.version, "Archived")
        client.transition_model_version_stage(model_name, version, "Production")
        client.set_model_version_tag(model_name, version, "registry_action", "promoted")
        print(
            f"  [OK] v{version} -> Production  "
            f"(MAE={mae:.4f}, prod v{prod_mv.version} had no MAE tag)"
        )
        return {"version": version, "mae": mae, "action": "promoted"}

    prod_mae = float(prod_mae_tag)

    # -- 6. Compare and promote or keep as challenger --------------------------
    if mae < prod_mae:
        # New model improves on Production - promote and archive the old
        client.transition_model_version_stage(model_name, prod_mv.version, "Archived")
        client.transition_model_version_stage(model_name, version, "Production")
        client.set_model_version_tag(model_name, version, "registry_action", "promoted")
        improvement = prod_mae - mae
        print(
            f"  [OK] v{version} -> Production  "
            f"(MAE={mae:.4f} < prod MAE={prod_mae:.4f}, delta={improvement:.4f})"
        )
        return {
            "version": version,
            "mae": mae,
            "action": "promoted",
            "prev_mae": prod_mae,
        }
    else:
        # New model does not improve - send to Staging as a challenger
        client.transition_model_version_stage(model_name, version, "Staging")
        client.set_model_version_tag(
            model_name, version, "registry_action", "challenger"
        )
        gap = mae - prod_mae
        print(
            f"  [--] v{version} -> Staging (challenger)  "
            f"(MAE={mae:.4f} >= prod MAE={prod_mae:.4f}, gap={gap:.4f})"
        )
        return {
            "version": version,
            "mae": mae,
            "action": "challenger",
            "prod_mae": prod_mae,
        }
