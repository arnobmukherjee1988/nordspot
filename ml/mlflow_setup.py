"""MLflow experiment setup for NordSpot.

Single source of truth for experiment names and tracking URI.
Every training module imports get_tracking_uri() and EXPERIMENTS from here.

Usage (one-time setup, idempotent):
    python -m ml.mlflow_setup

Typical import in training scripts:
    from ml.mlflow_setup import get_tracking_uri, EXPERIMENTS
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXPERIMENTS["lgbm"])
"""

from __future__ import annotations

import os

import mlflow

# ── Canonical experiment names ────────────────────────────────────────────────
# Keyed by short model alias so training scripts reference EXPERIMENTS["lgbm"]
# rather than hard-coding string literals that could drift across files.
EXPERIMENTS: dict[str, str] = {
    "lgbm": "nordspot-lgbm",
    "xgboost": "nordspot-xgboost",
    "catboost": "nordspot-catboost",
    "lear": "nordspot-lear",
    "ensemble": "nordspot-ensemble",
}

_DEFAULT_TRACKING_URI = "http://localhost:5000"


def get_tracking_uri() -> str:
    """Return MLflow tracking URI from env or fall back to localhost:5000."""
    return os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)


def create_experiments(
    tracking_uri: str | None = None,
) -> dict[str, str]:
    """Create all NordSpot experiments if they do not already exist.

    Idempotent — safe to run multiple times; existing experiments are left
    untouched and their IDs are returned alongside any newly created ones.

    Args:
        tracking_uri: Override the tracking URI (defaults to get_tracking_uri()).

    Returns:
        Mapping of model alias → MLflow experiment ID string.
    """
    uri = tracking_uri or get_tracking_uri()
    mlflow.set_tracking_uri(uri)

    ids: dict[str, str] = {}
    for alias, name in EXPERIMENTS.items():
        existing = mlflow.get_experiment_by_name(name)
        if existing is None:
            exp_id = mlflow.create_experiment(name)
            print(f"  [OK] Created experiment '{name}'  (id={exp_id})")
        else:
            exp_id = existing.experiment_id
            print(f"  [--] Experiment '{name}' already exists (id={exp_id})")
        ids[alias] = exp_id

    return ids


if __name__ == "__main__":
    print(f"Connecting to MLflow at {get_tracking_uri()} ...")
    result = create_experiments()
    print(f"\nDone — {len(result)} experiments ready.")
