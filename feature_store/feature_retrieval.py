"""NordSpot Feast feature retrieval helpers.

Wraps Feast's get_historical_features (training) and get_online_features
(serving) behind a stable interface so callers never import feast directly.

Usage — training (Epic 4):
    from feast.feature_retrieval import get_training_features

    entity_df = pd.DataFrame({
        "zone": ["10Y1001A1001A46L"] * n,
        "valid_time": timestamps,      # UTC, hourly
        "price": labels,               # target variable
    })
    features_df = get_training_features(entity_df)

Usage — serving (Epic 5):
    from feast.feature_retrieval import get_online_features

    row = get_online_features(zones=["10Y1001A1001A46L"])
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from feast import FeatureStore

_REPO_PATH = Path(__file__).resolve().parent  # feast/ directory


def _store() -> FeatureStore:
    """Return a FeatureStore connected to the NordSpot feature repo."""
    return FeatureStore(repo_path=str(_REPO_PATH))


def get_training_features(
    entity_df: pd.DataFrame,
    feature_service_name: str = "nordspot_forecast_features",
) -> pd.DataFrame:
    """Retrieve point-in-time correct historical features for training.

    Feast joins features to each (zone, valid_time) row using only data that
    was available at that timestamp — no temporal leakage.

    Parameters
    ----------
    entity_df:
        DataFrame with at minimum columns ``zone`` (str, EIC code) and
        ``valid_time`` (datetime, UTC-aware). May also include the ``price``
        target column, which Feast passes through unchanged.
    feature_service_name:
        Name of the registered FeatureService. Default is the full NordSpot set.

    Returns
    -------
    pd.DataFrame
        entity_df left-joined with all feature columns — ready for model.fit().
    """
    store = _store()
    job = store.get_historical_features(
        entity_df=entity_df,
        features=store.get_feature_service(feature_service_name),
    )
    return job.to_df()


def get_online_features(
    zones: list[str],
    feature_service_name: str = "nordspot_forecast_features",
) -> pd.DataFrame:
    """Retrieve the latest materialised feature values for a list of zones.

    Called at serving time by the FastAPI app (Epic 5). Reads from the SQLite
    online store (local) or Redis (production — Epic 7).

    Parameters
    ----------
    zones:
        List of ENTSO-E EIC codes, e.g. ``["10Y1001A1001A46L"]`` for SE3.
    feature_service_name:
        Name of the registered FeatureService.

    Returns
    -------
    pd.DataFrame
        One row per zone with the latest feature values from the online store.
    """
    store = _store()
    response = store.get_online_features(
        features=store.get_feature_service(feature_service_name),
        entity_rows=[{"zone": z} for z in zones],
    )
    return response.to_df()
