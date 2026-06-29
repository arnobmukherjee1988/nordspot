"""LightGBM quantile regression model for SE3 day-ahead price forecasting.

v2 improvements over baseline:
    - num_leaves 63 -> 127  (more expressive trees)
    - min_child_samples 20 -> 50  (key anti-overfitting guard)
    - colsample_bytree 0.8 -> 0.6  (stronger feature-level regularisation)
    - subsample 0.8 -> 0.7  (stronger row-level regularisation)
    - reg_alpha / reg_lambda 0.1 -> 0.3  (L1 + L2 penalty increase)
    - learning_rate 0.05 -> 0.03  (slower, finer convergence)
    - n_estimators 1000 -> 3000 with early stopping on held-out 15 % window
    - Recency sample weights: exponential decay, half-life = 365 days
    - Extended FEATURE_COLS: 7 price lags, 2 rolling stats,
      2 calendar interactions, 2 weather interactions
    - Split conformal calibration: post-hoc interval correction guaranteeing
      TARGET_COVERAGE marginal coverage on any future exchange of data
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
MODEL_DIR.mkdir(exist_ok=True)

QUANTILES = {"q05": 0.05, "q50": 0.50, "q95": 0.95}

# -- Feature columns -----------------------------------------------------------
# Ordered from most to least important (heuristic) for readability.
# ALL columns must exist in the DataFrame returned by pipeline.features.build_features.
FEATURE_COLS: list[str] = [
    # Price lags - 7 lags covering 1 week + cross-hour context + 2 weeks
    "price_lag24h",  # same hour yesterday          (core AR feature)
    "price_lag48h",  # same hour 2 days ago
    "price_lag168h",  # same hour 1 week ago         (weekly seasonality)
    "price_lag72h",  # same hour 3 days ago         (Mon<->Thu pattern)
    "price_lag336h",  # same hour 2 weeks ago        (fortnightly outage cycle)
    "price_lag23h",  # adjacent-hour yesterday (-1 h) - cross-hour context
    "price_lag25h",  # adjacent-hour yesterday (+1 h) - cross-hour context
    # Rolling statistics (computed on lag-24h price -> no leakage)
    "price_roll24h",  # 24 h rolling mean
    "price_roll168h",  # 168 h rolling mean
    # Calendar
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_of_week",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    # Calendar interactions
    "hour_x_month",  # captures seasonal time-of-day price shape
    "weekend_x_hour",  # weekend vs weekday daily profile
    # Weather
    "temperature",
    "wind_speed",
    "irradiance",
    # Weather interactions
    "temp_x_wind",  # cold + windy -> heating demand spike
    "temp_x_hour",  # morning temperature ramp-up effect
]

# -- Hyperparameters -----------------------------------------------------------
_LGB_PARAMS_BASE: dict = {
    "objective": "quantile",
    "metric": "quantile",
    # Tree structure - larger leaves offset by stronger leaf-count controls
    "num_leaves": 127,  # was 63; more expressive without depth explosion
    "min_child_samples": 50,  # was 20; each leaf needs >=50 samples -> key guard
    # Stochastic regularisation
    "subsample": 0.7,  # was 0.8; row-level bagging
    "colsample_bytree": 0.6,  # was 0.8; feature-level bagging per tree
    "subsample_freq": 1,  # apply row bagging every tree
    # Penalty regularisation
    "reg_alpha": 0.3,  # was 0.1; L1 - pushes small weights to 0
    "reg_lambda": 0.3,  # was 0.1; L2 - shrinks all weights
    # Learning
    "n_estimators": 3000,  # ceiling; actual count set by early stopping
    "learning_rate": 0.03,  # was 0.05; slower learning -> finer convergence
    "n_jobs": -1,
    "verbose": -1,
}

# Early stopping: hold out last VAL_FRAC of training rows (time-ordered)
VAL_FRAC = 0.15
EARLY_STOP_N = 50  # rounds without improvement before stopping

# Recency weighting: exponential decay so recent hours matter more
# Half-life of 365 days -> data from 1 year ago has 50 % weight of today
WEIGHT_HALF_LIFE_HOURS = 365 * 24

# -- Conformal calibration -----------------------------------------------------
# Split conformal prediction widens [q05, q95] by a data-driven correction c_hat
# so that the marginal coverage guarantee holds for any future test point.
# Reference: Angelopoulos & Bates (2021) "A gentle introduction to conformal
#            prediction and distribution-free uncertainty quantification."
TARGET_COVERAGE = 0.90  # desired marginal coverage
CONFORMAL_PATH = MODEL_DIR / "lgbm_conformal.pkl"


def _model_path(quantile_name: str) -> Path:
    return MODEL_DIR / f"lgbm_{quantile_name}.pkl"


def _recency_weights(n: int) -> np.ndarray:
    """Exponential decay weights, newest sample has weight ~=1, oldest ~= e^(-T/HL)."""
    decay = np.exp(-np.log(2) * np.arange(n)[::-1] / WEIGHT_HALF_LIFE_HOURS)
    # Normalise so mean weight = 1 (keeps loss scale comparable to unweighted)
    return decay / decay.mean()


def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return (X, y) - rows with any NaN feature OR NaN label dropped.

    Used for training only. For inference use _prep_features_only().
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df["price"] if "price" in df.columns else None

    valid = X.notna().all(axis=1)
    X = X[valid]
    if y is not None:
        y = y[valid]
        labelled = y.notna()
        X = X[labelled]
        y = y[labelled]

    return X, y


def _prep_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return X with rows that have any NaN feature dropped.

    Used for inference: does NOT filter by label (price is NaN for future dates).
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    valid = X.notna().all(axis=1)
    return X[valid]


def train(
    df: pd.DataFrame,
    verbose: bool = True,
) -> dict[str, lgb.LGBMRegressor]:
    """Fit one LGBMRegressor per quantile with early stopping and recency weights.

    Args:
        df:      Feature matrix from pipeline.features.build_features().
        verbose: Print progress per quantile.

    Returns:
        Dict mapping quantile name -> fitted model.
    """
    X, y = _prep(df)
    if y is None or len(y) == 0:
        raise ValueError("No labelled rows - 'price' column is all NaN.")

    n = len(X)
    n_val = max(24 * 30, int(n * VAL_FRAC))  # at least 30 days of validation
    n_tr = n - n_val

    X_tr, X_val = X.iloc[:n_tr], X.iloc[n_tr:]
    y_tr, y_val = y.iloc[:n_tr], y.iloc[n_tr:]

    # Recency weights applied only on training portion
    w_tr = _recency_weights(n_tr)

    models = {}
    for name, alpha in QUANTILES.items():
        params = {**_LGB_PARAMS_BASE, "alpha": alpha}
        model = lgb.LGBMRegressor(**params)

        model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOP_N, verbose=False),
                lgb.log_evaluation(-1),  # silence per-round output
            ],
        )

        with open(_model_path(name), "wb") as f:
            pickle.dump(model, f)
        models[name] = model

        if verbose:
            best = model.best_iteration_ or model.n_estimators
            print(
                f"  [OK] lgbm_{name}  "
                f"n_train={n_tr}  n_val={n_val}  best_iter={best}"
            )

    return models


def calibrate(
    actuals: pd.Series,
    q05_preds: pd.Series,
    q95_preds: pd.Series,
    target_coverage: float = TARGET_COVERAGE,
) -> float:
    """Fit a split-conformal correction on holdout predictions and save it.

    Algorithm (Angelopoulos & Bates 2021, Section2):
      1. Compute nonconformity score for each holdout point:
            s_i = max(q05_i - y_i,  y_i - q95_i)
         s_i > 0 when y_i falls outside [q05_i, q95_i]; <= 0 when inside.
      2. Find c_hat = ceil((n+1)(1-alpha))/n quantile of {s_1,...,s_n}.
      3. Calibrated interval: [q05 - c_hat,  q95 + c_hat].

    This gives a finite-sample marginal coverage guarantee:
        P(y_{n+1} in [q_hat05 - c_hat, q_hat95 + c_hat]) >= 1 - alpha

    Note: calibration and metric reporting use the same holdout - a common
    practical trade-off.  A separate calibration split would be strictly
    cleaner but is impractical given the 90-day holdout size.

    Args:
        actuals:         Observed prices on the holdout (pd.Series, UTC index).
        q05_preds:       lgbm_q05 column from predict() on the same period.
        q95_preds:       lgbm_q95 column from predict() on the same period.
        target_coverage: Desired marginal coverage (default 0.90).

    Returns:
        c_hat - the correction in EUR/MWh (also saved to lgbm_conformal.pkl).
    """
    # Align on shared non-NaN index
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

    # Nonconformity scores: distance outside the interval (negative = inside)
    scores = np.maximum(lo - y, y - hi)

    # Finite-sample quantile level
    alpha = 1.0 - target_coverage
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    correction = float(np.quantile(scores, level))

    # Pre-calibration coverage (for reporting)
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
        f"  [OK] Conformal calibration:  "
        f"raw coverage {raw_coverage:.1%} -> target {target_coverage:.0%}  "
        f"| c^ = {correction:+.3f} EUR/MWh  (n_cal = {n:,})"
    )
    return correction


def _load_conformal_correction() -> float | None:
    """Return saved conformal correction c_hat, or None if not yet calibrated."""
    if not CONFORMAL_PATH.exists():
        return None
    with open(CONFORMAL_PATH, "rb") as f:
        return pickle.load(f)["correction"]


def predict(df: pd.DataFrame, apply_conformal: bool = True) -> pd.DataFrame:
    """Load saved models and return quantile forecasts.

    Args:
        df:               Feature matrix from pipeline.features.build_features().
        apply_conformal:  If True (default) and lgbm_conformal.pkl exists, the
                          q05/q95 interval is widened by c_hat to achieve
                          TARGET_COVERAGE marginal coverage.  Pass False when
                          predicting *for* calibration - otherwise the old
                          correction is baked into the inputs and c_hat collapses
                          to 0 on every subsequent training run.

    The q50 point forecast is never modified.
    """
    X = _prep_features_only(df)
    out = pd.DataFrame(index=df.index)

    for name in QUANTILES:
        path = _model_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Run train() first.")
        with open(path, "rb") as f:
            model = pickle.load(f)
        col = f"lgbm_{name}"
        out[col] = np.nan
        if not X.empty:  # guard: short windows produce empty X
            out.loc[X.index, col] = model.predict(X)

    # Apply conformal correction to widen intervals if calibration file exists
    if apply_conformal:
        c = _load_conformal_correction()
        if c is not None and c > 0:
            out["lgbm_q05"] = out["lgbm_q05"] - c
            out["lgbm_q95"] = out["lgbm_q95"] + c

    return out


def feature_importance(top_n: int = 15) -> pd.DataFrame:
    """Averaged feature importance (gain) across all three quantile models."""
    records = []
    for name in QUANTILES:
        path = _model_path(name)
        if not path.exists():
            continue
        with open(path, "rb") as f:
            model = pickle.load(f)
        # Use the actual feature names the model was trained on
        cols = model.feature_name_
        imp = pd.Series(model.feature_importances_, index=cols, name=name)
        records.append(imp)

    if not records:
        raise FileNotFoundError("No trained models found. Run train() first.")

    fi = pd.concat(records, axis=1)
    fi["mean"] = fi.mean(axis=1)
    return fi.sort_values("mean", ascending=False).head(top_n)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 8000
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")

    fake = pd.DataFrame(
        {col: rng.standard_normal(n) for col in FEATURE_COLS},
        index=idx,
    )
    fake.index.name = "valid_time"
    fake["price"] = (
        50
        + 10 * fake["hour_sin"]
        + 5 * fake["temperature"]
        + rng.standard_normal(n) * 15
    )

    print("Training LightGBM on synthetic data (with early stopping) ...")
    train(fake)
    preds = predict(fake)
    print(preds.describe().round(2))
    print("\nFeature importance (top 10):")
    print(feature_importance(10)[["mean"]].round(1).to_string())
