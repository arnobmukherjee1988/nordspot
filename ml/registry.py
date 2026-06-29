"""MLflow Model Registry: automatic promotion of the NordSpot ensemble model.

After each training run, register_and_promote() compares the new ensemble's MAE
against the current champion and promotes automatically:

    new MAE < champion MAE  ->  new version gets 'champion' alias, old loses it
    new MAE >= champion MAE ->  new version gets 'challenger' alias
    no champion yet         ->  new version -> champion unconditionally

Uses MLflow aliases (not deprecated stages). Alias mapping:
    champion   - current serving model (replaces Production stage)
    challenger - better than nothing, but not beating champion (replaces Staging)

The MAE is written as a version tag so comparisons are self-contained.
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

ALIAS_CHAMPION = "champion"
ALIAS_CHALLENGER = "challenger"


def _get_champion(client: MlflowClient, model_name: str):
    """Return the current champion ModelVersion, or None."""
    try:
        return client.get_model_version_by_alias(model_name, ALIAS_CHAMPION)
    except mlflow.exceptions.MlflowException:
        return None


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
        5. Fetch the current champion and compare MAEs.
        6. Assign 'champion' or 'challenger' alias accordingly.

    Args:
        run_id:        MLflow run ID from the ensemble training block.
        mae:           Test-window MAE for the new model (lower is better).
        model_name:    Registered model name (default "nordspot-ensemble").
        artifact_path: Artifact path logged inside the run.
        tracking_uri:  Override tracking URI (default: env / localhost:5000).

    Returns:
        dict with keys:
            version  (str)   - new version number
            mae      (float) - MAE of the new version
            action   (str)   - "promoted" | "challenger"
            prev_mae (float, optional) - MAE of the prior champion
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

    # -- 5. Fetch current champion (if any) ------------------------------------
    champion = _get_champion(client, model_name)

    if champion is None:
        # No champion yet - promote unconditionally
        client.set_registered_model_alias(model_name, ALIAS_CHAMPION, version)
        client.set_model_version_tag(model_name, version, "registry_action", "promoted")
        print(f"  [OK] v{version} -> champion  " f"(MAE={mae:.4f}, no prior champion)")
        return {"version": version, "mae": mae, "action": "promoted"}

    champ_mae_tag = champion.tags.get("mae")

    if champ_mae_tag is None:
        # Champion exists but has no MAE tag - treat as unknown quality, promote.
        client.delete_registered_model_alias(model_name, ALIAS_CHAMPION)
        client.set_registered_model_alias(model_name, ALIAS_CHAMPION, version)
        client.set_model_version_tag(model_name, version, "registry_action", "promoted")
        print(
            f"  [OK] v{version} -> champion  "
            f"(MAE={mae:.4f}, prior champion v{champion.version} had no MAE tag)"
        )
        return {"version": version, "mae": mae, "action": "promoted"}

    champ_mae = float(champ_mae_tag)

    # -- 6. Compare and promote or keep as challenger --------------------------
    if mae < champ_mae:
        # New model improves on champion - reassign alias
        client.delete_registered_model_alias(model_name, ALIAS_CHAMPION)
        client.set_registered_model_alias(model_name, ALIAS_CHAMPION, version)
        client.set_model_version_tag(model_name, version, "registry_action", "promoted")
        improvement = champ_mae - mae
        print(
            f"  [OK] v{version} -> champion  "
            f"(MAE={mae:.4f} < champion MAE={champ_mae:.4f}, delta={improvement:.4f})"
        )
        return {
            "version": version,
            "mae": mae,
            "action": "promoted",
            "prev_mae": champ_mae,
        }
    else:
        # New model does not improve - assign challenger alias
        client.set_registered_model_alias(model_name, ALIAS_CHALLENGER, version)
        client.set_model_version_tag(
            model_name, version, "registry_action", "challenger"
        )
        gap = mae - champ_mae
        print(
            f"  [--] v{version} -> challenger  "
            f"(MAE={mae:.4f} >= champion MAE={champ_mae:.4f}, gap={gap:.4f})"
        )
        return {
            "version": version,
            "mae": mae,
            "action": "challenger",
            "prod_mae": champ_mae,
        }
