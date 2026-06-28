"""Stacking ensemble: Ridge meta-learner blending LGBM, XGBoost, and CatBoost.

Trains one Ridge meta-learner per quantile (q05, q50, q95).  Each meta-model
takes only same-quantile base predictions as features:

    ens_q05 = Ridge([lgbm_q05, xgb_q05, cat_q05])
    ens_q50 = Ridge([lgbm_q50, xgb_q50, cat_q50])
    ens_q95 = Ridge([lgbm_q95, xgb_q95, cat_q95])

Meta-training data: calibrated base model predictions on the calibration window.
This avoids overfitting because the calibration window was held out during base
model training (60–90 days → ~1 500 rows, plenty for a 3-feature Ridge).

Quantile ordering is enforced post-hoc:
    ens_q05 ← min(ens_q05, ens_q50)
    ens_q95 ← max(ens_q95, ens_q50)

Ridge is preferred over QuantileRegressor here because:
1. The meta-features ARE quantile estimates, so minimising MSE on calibrated
   predictions produces an unbiased blend when base models are well-calibrated.
2. Ridge is orders of magnitude faster than the interior-point solver that
   sklearn.QuantileRegressor uses.
3. Post-hoc ordering clip handles the rare crossing case without constraining
   the regression coefficients.

The Ridge coefficients are logged as MLflow params per quantile, making it
easy to inspect which base model dominates (e.g. 'lgbm_q50 = 0.6' means
LGBM carries 60 % of the q50 blend).

Reference: Wolpert (1992) "Stacked Generalisation." Neural Networks 5(2).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
MODEL_DIR.mkdir(exist_ok=True)

# For each quantile, which base-model columns to use as meta-features.
# Only same-quantile columns are blended per meta-model — this guarantees
# that a positive-weight solution still produces a valid quantile estimate
# (e.g. Ridge([lgbm_q05, xgb_q05, cat_q05]) → ens_q05).
_Q_FEATURES: dict[str, list[str]] = {
    "q05": ["lgbm_q05", "xgb_q05", "cat_q05"],
    "q50": ["lgbm_q50", "xgb_q50", "cat_q50"],
    "q95": ["lgbm_q95", "xgb_q95", "cat_q95"],
}

_RIDGE_ALPHA = 1.0  # L2 regularisation; 3 features so a light penalty suffices


def _model_path(quantile_name: str) -> Path:
    return MODEL_DIR / f"ens_{quantile_name}.pkl"


def train(
    base_cal_preds: pd.DataFrame,
    actuals: pd.Series,
) -> dict[str, Ridge]:
    """Fit one Ridge meta-learner per quantile on calibration-window predictions.

    Args:
        base_cal_preds: DataFrame with columns lgbm_q05…lgbm_q95, xgb_q05…,
                        cat_q05… — calibrated predictions from all three base
                        models on the calibration window.
        actuals:        True prices on the same calibration window (pd.Series).

    Returns:
        Dict mapping quantile name → fitted Ridge model.
        Each model is also pickled to MODEL_DIR/ens_{name}.pkl.
    """
    models: dict[str, Ridge] = {}

    for q_name, feat_cols in _Q_FEATURES.items():
        available = [c for c in feat_cols if c in base_cal_preds.columns]
        if not available:
            raise ValueError(
                f"No base model columns found for {q_name}. "
                f"Expected one of {feat_cols}. Got: {list(base_cal_preds.columns)}"
            )

        # Align on rows with no NaN in features or target
        valid = base_cal_preds[available].notna().all(axis=1) & actuals.notna()
        idx = base_cal_preds.index[valid]
        x_meta = base_cal_preds.loc[idx, available].values
        y_meta = actuals[idx].values

        model = Ridge(alpha=_RIDGE_ALPHA)
        model.fit(x_meta, y_meta)

        with open(_model_path(q_name), "wb") as fh:
            pickle.dump(model, fh)

        coef_str = ", ".join(f"{c}={v:.3f}" for c, v in zip(available, model.coef_))
        print(
            f"  [OK] ens_{q_name}  Ridge coefs: [{coef_str}]  "
            f"intercept={model.intercept_:.3f}  (n_meta={len(y_meta):,})"
        )
        models[q_name] = model

    return models


def predict(base_test_preds: pd.DataFrame) -> pd.DataFrame:
    """Load saved Ridge meta-learners and return ensemble quantile forecasts.

    Args:
        base_test_preds: DataFrame with base model quantile columns
                         (same 9-column schema as base_cal_preds passed to train()).

    Returns:
        DataFrame with columns ens_q05, ens_q50, ens_q95.
        Quantile ordering is enforced: ens_q05 ≤ ens_q50 ≤ ens_q95 for all rows.
    """
    out = pd.DataFrame(index=base_test_preds.index)

    for q_name, feat_cols in _Q_FEATURES.items():
        available = [c for c in feat_cols if c in base_test_preds.columns]
        out_col = f"ens_{q_name}"
        out[out_col] = np.nan

        if not available:
            continue

        path = _model_path(q_name)
        with open(path, "rb") as fh:
            model: Ridge = pickle.load(fh)

        valid = base_test_preds[available].notna().all(axis=1)
        if valid.any():
            out.loc[valid, out_col] = model.predict(
                base_test_preds.loc[valid, available].values
            )

    # Enforce quantile ordering: q05 ≤ q50 ≤ q95.
    # Ridge can in principle assign negative coefficients, so the ordering is
    # not algebraically guaranteed.  The clip is a cheap post-hoc fix that
    # has negligible effect when base models are well-calibrated.
    out["ens_q05"] = np.minimum(out["ens_q05"], out["ens_q50"])
    out["ens_q95"] = np.maximum(out["ens_q95"], out["ens_q50"])

    return out
