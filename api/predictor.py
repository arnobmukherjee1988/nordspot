"""Model inference pipeline — called by the forecast router on every request.

Runs the full prediction chain:
    1. LGBM, XGBoost, CatBoost → per-model quantile predictions (q05/q50/q95)
    2. Ridge ensemble meta-learners → blended ens_q05, ens_q50, ens_q95

Why disk-based models rather than store.model.predict()?
    MLflow only registers the q50 Ridge meta-learner ("ensemble_q50" artefact
    — see ml/registry.py and ml/train.py line 885).  The q05 and q95 Ridge
    models are saved to MODEL_DIR during training but are not registered.
    Using ml.models.ensemble.predict() covers all three quantiles via the
    same code path used during training evaluation, avoiding quantile skew.

    store.model / store.is_ready remains the authoritative gate that tells
    the API whether training has completed and a Production model exists in
    the registry.  Once a full PyFunc wrapper is logged (a future story), this
    module can be replaced with a single store.model.predict() call.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("nordspot.api.predictor")


def run_inference(features_df: pd.DataFrame) -> pd.DataFrame:
    """Run the full prediction chain on a 24-row feature DataFrame.

    Parameters
    ----------
    features_df:
        Feature matrix as returned by ``api.features.get_inference_features()``.
        Must contain all columns expected by the base models (price lags,
        calendar features, weather, load, generation, net exchange).

    Returns
    -------
    pd.DataFrame
        24 rows with columns ``ens_q05``, ``ens_q50``, ``ens_q95`` (EUR/MWh).
        Row order matches the input.  Quantile ordering is guaranteed
        (q05 ≤ q50 ≤ q95) by ml.models.ensemble.predict().
    """
    from ml.models import catboost as cat_model
    from ml.models import ensemble as ens_model
    from ml.models import lgbm
    from ml.models import xgboost as xgb_model

    logger.info("Running base model inference on %d rows", len(features_df))

    # ── 1. Base model predictions ─────────────────────────────────────────
    # Each predict() loads saved pickle files from MODEL_DIR and returns a
    # DataFrame with zone-prefixed column names (lgbm_q05, lgbm_q50, lgbm_q95 …).
    lgbm_preds = lgbm.predict(features_df)  # lgbm_q05, lgbm_q50, lgbm_q95
    xgb_preds = xgb_model.predict(features_df)  # xgb_q05,  xgb_q50,  xgb_q95
    cat_preds = cat_model.predict(features_df)  # cat_q05,  cat_q50,  cat_q95

    base_preds = pd.concat([lgbm_preds, xgb_preds, cat_preds], axis=1)

    # ── 2. Ensemble Ridge blending ────────────────────────────────────────
    # ens_model.predict() loads ens_q05.pkl / ens_q50.pkl / ens_q95.pkl,
    # blends one quantile per Ridge model, then enforces ordering.
    ens_preds = ens_model.predict(base_preds)  # ens_q05, ens_q50, ens_q95

    logger.info(
        "Inference complete — q50 range: %.1f – %.1f EUR/MWh",
        float(ens_preds["ens_q50"].min()),
        float(ens_preds["ens_q50"].max()),
    )
    return ens_preds
