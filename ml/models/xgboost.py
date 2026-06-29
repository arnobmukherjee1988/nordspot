"""XGBoost quantile regression model for NordSpot day-ahead price forecasting.

Mirrors the interface of ml/models/lgbm.py exactly:
    train(df)    -> dict[str, XGBRegressor]
    predict(df)  -> pd.DataFrame with columns xgb_q05, xgb_q50, xgb_q95
    calibrate(...) -> float (split-conformal correction c_hat)

Reference: Marcjasz et al. (2023) "Distributional neural networks for
electricity price forecasting." Energy Economics 106, 105742 - confirms
XGBoost as a competitive tree-based baseline for day-ahead price forecasting.

Key differences from LightGBM:
    - Uses XGBoost's native quantile objective ("reg:quantileerror") rather
      than a pinball-loss approximation - guaranteed convex quantile loss.
    - max_depth + min_child_weight instead of num_leaves + min_child_samples.
    - early_stopping_rounds passed to XGBRegressor constructor (XGBoost >=2.0).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Reuse the shared feature list and recency-weight helper from the LGBM module
# to guarantee XGBoost is trained on exactly the same feature set.
from ml.models.lgbm import FEATURE_COLS, _recency_weights

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
MODEL_DIR.mkdir(exist_ok=True)

QUANTILES: dict[str, float] = {"q05": 0.05, "q50": 0.50, "q95": 0.95}

# -- Hyperparameters -----------------------------------------------------------
_XGB_PARAMS_BASE: dict = {
    "tree_method": "hist",  # approximate histogram - fast on CPU
    "max_depth": 6,  # analogous to num_leaves=127 in LGBM
    "min_child_weight": 50,  # mirrors lgbm.min_child_samples
    "subsample": 0.7,  # row-level bagging
    "colsample_bytree": 0.6,  # feature-level bagging
    "reg_alpha": 0.3,  # L1 penalty - mirrors lgbm.reg_alpha
    "reg_lambda": 0.3,  # L2 penalty - mirrors lgbm.reg_lambda
    "learning_rate": 0.03,  # mirrors lgbm.learning_rate
    "n_estimators": 3000,  # ceiling; early stopping sets true count
    "n_jobs": -1,
    "verbosity": 0,
}

VAL_FRAC = 0.15  # held-out fraction for early stopping
EARLY_STOP_N = 50  # rounds without improvement before stopping
TARGET_COVERAGE = 0.90  # desired marginal coverage for conformal calibration
CONFORMAL_PATH = MODEL_DIR / "xgb_conformal.pkl"


def _model_path(quantile_name: str) -> Path:
    return MODEL_DIR / f"xgb_{quantile_name}.pkl"


def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return (X, y) dropping rows with any NaN feature."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    x = df[available].copy()
    y = df["price"] if "price" in df.columns else None

    valid = x.notna().all(axis=1)
    x = x[valid]
    if y is not None:
        y = y[valid]
        labelled = y.notna()
        x = x[labelled]
        y = y[labelled]

    return x, y


def train(
    df: pd.DataFrame,
    verbose: bool = True,
) -> dict[str, xgb.XGBRegressor]:
    """Fit one XGBRegressor per quantile with early stopping and recency weights.

    Args:
        df:      Feature matrix from pipeline.features.build_features().
        verbose: Print progress per quantile.

    Returns:
        Dict mapping quantile name -> fitted model.
    """
    x, y = _prep(df)
    if y is None or len(y) == 0:
        raise ValueError("No labelled rows - 'price' column is all NaN.")

    n = len(x)
    # Cap n_val at n//2 so n_tr is always positive even on small datasets.
    # In production n is always >> 720, so this guard only fires in unit tests.
    n_val = min(max(24 * 30, int(n * VAL_FRAC)), n // 2)
    n_tr = n - n_val

    x_tr, x_val = x.iloc[:n_tr], x.iloc[n_tr:]
    y_tr, y_val = y.iloc[:n_tr], y.iloc[n_tr:]
    w_tr = _recency_weights(n_tr)

    models = {}
    for name, alpha in QUANTILES.items():
        model = xgb.XGBRegressor(
            **_XGB_PARAMS_BASE,
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            early_stopping_rounds=EARLY_STOP_N,
        )
        model.fit(
            x_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        with open(_model_path(name), "wb") as f:
            pickle.dump(model, f)
        models[name] = model

        if verbose:
            best = (
                model.best_iteration
                if model.best_iteration is not None
                else model.n_estimators
            )
            print(
                f"  [OK] xgb_{name}  "
                f"n_train={n_tr}  n_val={n_val}  best_iter={best}"
            )

    return models


def calibrate(
    actuals: pd.Series,
    q05_preds: pd.Series,
    q95_preds: pd.Series,
    target_coverage: float = TARGET_COVERAGE,
) -> float:
    """Fit split-conformal correction on holdout predictions and save it.

    Identical algorithm to lgbm.calibrate - see that docstring for derivation.
    """
    idx = (
        actuals.dropna()
        .index.intersection(q05_preds.dropna().index)
        .intersection(q95_preds.dropna().index)
    )
    if len(idx) == 0:
        raise ValueError("No overlapping non-NaN rows - cannot calibrate.")

    y = actuals[idx].values
    lo = q05_preds[idx].values
    hi = q95_preds[idx].values
    n = len(y)

    scores = np.maximum(lo - y, y - hi)
    alpha = 1.0 - target_coverage
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    correction = float(np.quantile(scores, level))

    raw_coverage = float(np.mean((y >= lo) & (y <= hi)))
    bundle = {
        "correction": correction,
        "n_calibration": n,
        "target_coverage": target_coverage,
        "raw_coverage": raw_coverage,
    }
    with open(CONFORMAL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(
        f"  [OK] XGBoost conformal calibration:  "
        f"raw coverage {raw_coverage:.1%} -> target {target_coverage:.0%}  "
        f"| c_hat = {correction:+.3f} EUR/MWh  (n_cal = {n:,})"
    )
    return correction


def _load_conformal_correction() -> float | None:
    """Return saved conformal correction c_hat, or None if not yet calibrated."""
    if not CONFORMAL_PATH.exists():
        return None
    with open(CONFORMAL_PATH, "rb") as f:
        return pickle.load(f)["correction"]


def predict(df: pd.DataFrame, apply_conformal: bool = True) -> pd.DataFrame:
    """Load saved XGBoost models and return quantile forecasts.

    Args:
        df:               Feature matrix from pipeline.features.build_features().
        apply_conformal:  Widen [q05, q95] by c_hat if calibration file exists.
                          Pass False when predicting for calibration itself.

    Returns:
        DataFrame with columns xgb_q05, xgb_q50, xgb_q95 (same index as df).
    """
    x, _ = _prep(df)
    out = pd.DataFrame(index=df.index)

    for name in QUANTILES:
        path = _model_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Run train() first.")
        with open(path, "rb") as f:
            model = pickle.load(f)
        col = f"xgb_{name}"
        out[col] = np.nan
        if not x.empty:
            out.loc[x.index, col] = model.predict(x)

    if apply_conformal:
        c = _load_conformal_correction()
        if c is not None and c > 0:
            out["xgb_q05"] = out["xgb_q05"] - c
            out["xgb_q95"] = out["xgb_q95"] + c

    return out
