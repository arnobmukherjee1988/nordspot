"""LEAR - LASSO Estimated AutoRegressive model for SE3 day-ahead price forecasting.

Reference: Lago et al. (2021) "Forecasting day-ahead electricity prices:
           A review of state-of-the-art algorithms, best practices and
           an open-access benchmark." Applied Energy 293, 116983.

v2 improvements over baseline:
    - LASSO alpha selected by LassoCV with TimeSeriesSplit(3) - was fixed at 0.001
    - Cross-hour AR lags 23 h and 25 h added - breaks the rigid hour-silo assumption
    - Residual quantiles computed on a rolling 365-day window - adapts to volatility
      regime changes rather than using the full in-sample residual history
    - Extended exogenous features consistent with the updated feature matrix
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
MODEL_DIR.mkdir(exist_ok=True)

# Rolling window for residual quantile estimation (hours).
# Using only the most recent RESID_WINDOW hours of residuals means the
# intervals adapt to current market volatility rather than averaging
# over a potentially stale historical distribution.
RESID_WINDOW = 365 * 24

# LassoCV candidate alphas - logarithmically spaced to span weak -> strong regularisation
_LASSO_ALPHAS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

# Exogenous (non-price) features used by LEAR
# Includes the two new interaction features added in features.py v2
_EXOG_COLS = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_holiday",
    "temperature",
    "wind_speed",
    "irradiance",
    "hour_x_month",  # v2 addition
    "weekend_x_hour",  # v2 addition
    "temp_x_hour",  # v2 addition
]

# AR lags used by LEAR.
# 23 h and 25 h are the adjacent hours from the previous day - they provide
# cross-hour context so models for adjacent hours are no longer completely
# independent (the "too rigid" problem in the baseline).
_AR_LAGS_COLS = [
    "price_lag23h",
    "price_lag24h",
    "price_lag25h",
    "price_lag48h",
    "price_lag72h",
    "price_lag168h",
    "price_lag336h",
    "price_roll24h",
    "price_roll168h",
]


def _model_path(hour: int) -> Path:
    return MODEL_DIR / f"lear_h{hour:02d}.pkl"


def _build_X(df: pd.DataFrame) -> pd.DataFrame:
    cols = _AR_LAGS_COLS + _EXOG_COLS
    available = [c for c in cols if c in df.columns]
    return df[available].copy()


def train(df: pd.DataFrame, verbose: bool = True) -> dict[int, tuple]:
    """Fit one LassoCV model per hour-of-day (24 models) and save to MODEL_DIR.

    For each hour:
      1. StandardScaler normalises features (required for LASSO).
      2. LassoCV selects optimal alpha via TimeSeriesSplit(3) CV.
      3. Residual quantiles are estimated on the most recent RESID_WINDOW
         hours of in-sample predictions (rolling window, not full history).

    Args:
        df:      Feature matrix from pipeline.features.build_features().
        verbose: Print summary on completion.

    Returns:
        Dict mapping hour -> (scaler, lasso_cv_model, rq05, rq95, best_alpha).
    """
    df = df.copy()
    if "valid_time" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("valid_time")
    df["_hour"] = df.index.hour

    trained = {}
    alpha_chosen = {}

    cv = TimeSeriesSplit(n_splits=3)

    for h in range(24):
        subset = df[df["_hour"] == h].dropna(subset=["price"])
        X_raw = _build_X(subset)
        y = subset["price"]

        valid = X_raw.notna().all(axis=1)
        X_raw, y = X_raw[valid], y[valid]

        if len(X_raw) < 60:
            if verbose:
                print(f"  [WARN] Hour {h:02d}: only {len(X_raw)} samples - skipping")
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # LassoCV: tests all candidate alphas with time-series cross-validation
        # n_jobs=-1 parallelises across alpha candidates
        model = LassoCV(
            alphas=_LASSO_ALPHAS,
            cv=cv,
            max_iter=10_000,
            tol=1e-4,
            n_jobs=-1,
        )
        model.fit(X, y)

        best_alpha = float(model.alpha_)
        alpha_chosen[h] = best_alpha

        # In-sample predictions for residual estimation
        resid = y.values - model.predict(X)

        # Rolling residual window - use only the most recent RESID_WINDOW hours
        # so that interval width adapts to current volatility regime
        window = resid[-RESID_WINDOW:] if len(resid) > RESID_WINDOW else resid
        rq05 = float(np.quantile(window, 0.05))
        rq95 = float(np.quantile(window, 0.95))

        bundle = (scaler, model, rq05, rq95, best_alpha)
        with open(_model_path(h), "wb") as f:
            pickle.dump(bundle, f)
        trained[h] = bundle

    if verbose:
        alphas = list(alpha_chosen.values())
        print(f"  [OK] Trained {len(trained)}/24 hourly LEAR models")
        print(
            f"       Alpha range: {min(alphas):.4f} - {max(alphas):.4f}  "
            f"(median {np.median(alphas):.4f})"
        )

    return trained


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Load saved LEAR models and return quantile forecasts."""
    df = df.copy()
    if "valid_time" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("valid_time")
    df["_hour"] = df.index.hour

    out = pd.DataFrame(
        index=df.index,
        columns=["lear_q05", "lear_q50", "lear_q95"],
        dtype=float,
    )

    for h in range(24):
        path = _model_path(h)
        if not path.exists():
            continue
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        # Support both old (4-tuple) and new (5-tuple) bundles
        scaler, model, rq05, rq95 = bundle[:4]

        subset = df[df["_hour"] == h]
        X_raw = _build_X(subset)
        valid = X_raw.notna().all(axis=1)
        X_raw = X_raw[valid]
        if X_raw.empty:
            continue

        X = scaler.transform(X_raw)
        p50 = model.predict(X)

        out.loc[X_raw.index, "lear_q50"] = p50
        out.loc[X_raw.index, "lear_q05"] = p50 + rq05
        out.loc[X_raw.index, "lear_q95"] = p50 + rq95

    return out


def alpha_summary() -> pd.Series:
    """Return the CV-selected alpha for each hour (diagnostic tool)."""
    rows = {}
    for h in range(24):
        path = _model_path(h)
        if not path.exists():
            continue
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        if len(bundle) == 5:
            rows[h] = bundle[4]
    return pd.Series(rows, name="best_alpha")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 8000
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")

    from ml.models.lgbm import FEATURE_COLS

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

    print("Training LEAR (LassoCV) on synthetic data ...")
    train(fake)

    print("\nAlpha selected per hour:")
    print(alpha_summary().to_string())

    preds = predict(fake)
    print("\nPrediction summary:")
    print(preds.describe().round(2))
