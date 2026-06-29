"""CatBoost quantile regression model for NordSpot day-ahead price forecasting.

Mirrors the interface of ml/models/lgbm.py and ml/models/xgboost.py:
    train(df)    -> dict[str, CatBoostRegressor]
    predict(df)  -> pd.DataFrame with columns cat_q05, cat_q50, cat_q95
    calibrate(...) -> float (split-conformal correction c_hat)

Key differentiator vs LGBM / XGBoost:
    CatBoost handles integer-valued categorical features natively via its
    target-statistic encoding.  Rather than one-hot encoding 'hour' (24 levels),
    'weekday' (7 levels), and 'month' (12 levels) before training, we declare
    them as cat_features and let CatBoost learn optimal per-fold mean-target
    encodings with ordered boosting - this typically yields +1-2 MAE improvement
    on data with strong diurnal and seasonal price patterns.

Reference: Prokhorenkova et al. (2018) "CatBoost: unbiased boosting with
categorical features." NeurIPS 31.  See also Lago et al. (2021) Appendix B
for quantile CatBoost benchmarks on European day-ahead markets.

Implementation notes:
    - loss_function = f"Quantile:alpha={alpha}" (not a constructor kwarg)
    - Models serialised with .save_model() / .load_model() in CatBoost's
      native binary format (.cbm) - faster than pickle for large forests.
    - Pool() pre-processes cat_features once per dataset, avoiding repeated
      encoding overhead inside fit/predict.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from ml.models.lgbm import FEATURE_COLS, _recency_weights

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
MODEL_DIR.mkdir(exist_ok=True)

QUANTILES: dict[str, float] = {"q05": 0.05, "q50": 0.50, "q95": 0.95}

# Columns treated as categoricals - integer-valued, discrete, strongly periodic.
# CatBoost learns target-statistic encodings instead of relying on one-hot or
# cyclical sin/cos transforms (those are kept too, so the model can use either).
CAT_FEATURE_COLS: list[str] = ["hour", "weekday", "month"]

# -- Hyperparameters -----------------------------------------------------------
_CB_PARAMS_BASE: dict = {
    "iterations": 3000,  # ceiling; early stopping sets true count
    "learning_rate": 0.03,  # mirrors lgbm / xgboost learning_rate
    "depth": 6,  # analogous to lgbm.num_leaves ~= 2^6
    "l2_leaf_reg": 3.0,  # L2 leaf regularisation (CatBoost default)
    "bagging_temperature": 0.5,  # Bayesian bootstrap intensity (0 = no bag)
    "random_strength": 1.0,  # feature importance perturbation on splits
    "border_count": 128,  # histogram bins for numerical features
    "thread_count": -1,  # use all cores
    "random_seed": 42,
    "verbose": 0,  # suppress per-iteration stdout
}

VAL_FRAC = 0.15
EARLY_STOP_N = 50
TARGET_COVERAGE = 0.90
CONFORMAL_PATH = MODEL_DIR / "cat_conformal.pkl"


def _model_path(quantile_name: str) -> Path:
    return MODEL_DIR / f"cat_{quantile_name}.cbm"


def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return (X, y) with cat_features cast to int and NaN rows dropped.

    Used for training only. For inference use _prep_features_only().
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    x = df[available].copy()
    y = df["price"] if "price" in df.columns else None

    # Cast categorical columns to int so CatBoost accepts them.
    # They may arrive as float64 after merges / reindex ops.
    for col in CAT_FEATURE_COLS:
        if col in x.columns:
            x[col] = x[col].astype(int)

    valid = x.notna().all(axis=1)
    x = x[valid]
    if y is not None:
        y = y[valid]
        labelled = y.notna()
        x = x[labelled]
        y = y[labelled]

    return x, y


def _prep_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return X with cat_features cast to int and NaN-feature rows dropped.

    Used for inference: does NOT filter by label (price is NaN for future dates).
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    x = df[available].copy()
    for col in CAT_FEATURE_COLS:
        if col in x.columns:
            x[col] = x[col].astype(int)
    valid = x.notna().all(axis=1)
    return x[valid]


def train(
    df: pd.DataFrame,
    verbose: bool = True,
) -> dict[str, CatBoostRegressor]:
    """Fit one CatBoostRegressor per quantile with early stopping and recency weights.

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
    n_val = min(max(24 * 30, int(n * VAL_FRAC)), n // 2)
    n_tr = n - n_val

    x_tr, x_val = x.iloc[:n_tr], x.iloc[n_tr:]
    y_tr, y_val = y.iloc[:n_tr], y.iloc[n_tr:]
    w_tr = _recency_weights(n_tr)

    # Identify which of our cat_features are actually present in x
    available_cat = [c for c in CAT_FEATURE_COLS if c in x.columns]

    train_pool = Pool(x_tr, y_tr, cat_features=available_cat, weight=w_tr)
    val_pool = Pool(x_val, y_val, cat_features=available_cat)

    models = {}
    for name, alpha in QUANTILES.items():
        model = CatBoostRegressor(
            **_CB_PARAMS_BASE,
            loss_function=f"Quantile:alpha={alpha}",
        )
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=EARLY_STOP_N,
            verbose=False,
        )
        model.save_model(str(_model_path(name)))
        models[name] = model

        if verbose:
            best = model.best_iteration_
            print(
                f"  [OK] cat_{name}  "
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

    Identical algorithm to lgbm.calibrate and xgboost.calibrate.
    """
    import pickle  # lazy import - only needed here and in load

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
        f"  [OK] CatBoost conformal calibration:  "
        f"raw coverage {raw_coverage:.1%} -> target {target_coverage:.0%}  "
        f"| c_hat = {correction:+.3f} EUR/MWh  (n_cal = {n:,})"
    )
    return correction


def _load_conformal_correction() -> float | None:
    """Return saved conformal correction c_hat, or None if not yet calibrated."""
    import pickle

    if not CONFORMAL_PATH.exists():
        return None
    with open(CONFORMAL_PATH, "rb") as f:
        return pickle.load(f)["correction"]


def predict(df: pd.DataFrame, apply_conformal: bool = True) -> pd.DataFrame:
    """Load saved CatBoost models and return quantile forecasts.

    Args:
        df:               Feature matrix from pipeline.features.build_features().
        apply_conformal:  Widen [q05, q95] by c_hat if calibration file exists.
                          Pass False when predicting for calibration itself.

    Returns:
        DataFrame with columns cat_q05, cat_q50, cat_q95 (same index as df).
    """
    x = _prep_features_only(df)
    out = pd.DataFrame(index=df.index)

    available_cat = [c for c in CAT_FEATURE_COLS if c in x.columns]

    for name in QUANTILES:
        path = _model_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Run train() first.")
        model = CatBoostRegressor()
        model.load_model(str(path))

        col = f"cat_{name}"
        out[col] = np.nan
        if not x.empty:
            pred_pool = Pool(x, cat_features=available_cat)
            out.loc[x.index, col] = model.predict(pred_pool)

    if apply_conformal:
        c = _load_conformal_correction()
        if c is not None and c > 0:
            out["cat_q05"] = out["cat_q05"] - c
            out["cat_q95"] = out["cat_q95"] + c

    return out
