"""Walk-forward evaluation with CRPS, MAE, and reliability diagnostics.

Walk-forward (rolling-window) protocol
---------------------------------------
Training window: ``train_days`` days ending at each fold's cut-off.
Test window:     next ``test_days`` days (day-ahead: predicting D+1 ... D+test_days).
Step size:       ``step_days`` days between folds.

Metrics computed per fold, then aggregated:
    - CRPS        (primary - lower is better; rewards calibrated uncertainty)
    - MAE p50     (point forecast error)
    - Spike MAE   (MAE conditioned on actual price > spike_threshold EUR/MWh)
    - Coverage    (fraction of actuals inside [q05, q95] interval - target ~90 %)
    - Interval width (mean q95 - q05)

Reliability diagram data is also returned for the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import properscoring as ps
from timedb import TimeDBClient

from pipeline.features import build_features

# -- Result containers ---------------------------------------------------------


@dataclass
class FoldResult:
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train: int
    n_test: int

    # Per-model metrics
    crps: dict[str, float] = field(default_factory=dict)
    mae: dict[str, float] = field(default_factory=dict)
    spike_mae: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    interval_width: dict[str, float] = field(default_factory=dict)

    # Raw test predictions (for reliability diagram)
    actuals: pd.Series | None = None
    preds: dict[str, pd.DataFrame] | None = None


@dataclass
class EvalResult:
    folds: list[FoldResult]
    summary: pd.DataFrame  # mean +/- std across folds, one row per model
    reliability: pd.DataFrame  # calibration data for reliability diagram


# -- CRPS helper ---------------------------------------------------------------


def _crps_quantile(
    actuals: np.ndarray,
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
) -> float:
    """Approximate CRPS from three quantiles using the pinball loss identity.

    Uses properscoring's crps_gaussian as a fallback-free approximation:
    fits a Gaussian from the quantile spread then evaluates CRPS exactly.
    This is a well-known approximation when only a few quantiles are available.
    """
    # Approximate mean and std from quantile spread
    mu = q50
    # IQR-based sigma: (q95 - q05) covers ~90% -> / 3.29 (z_{0.95} - z_{0.05})
    sigma = np.clip((q95 - q05) / 3.29, a_min=1e-3, a_max=None)
    return float(np.mean(ps.crps_gaussian(actuals, mu=mu, sig=sigma)))


# -- Metrics -------------------------------------------------------------------


def _compute_metrics(
    actuals: pd.Series,
    forecasts: dict[str, pd.DataFrame],
    spike_threshold: float = 100.0,
) -> dict[str, dict[str, float]]:
    """Return per-model metric dicts for one fold."""
    results = {
        "crps": {},
        "mae": {},
        "spike_mae": {},
        "coverage": {},
        "interval_width": {},
    }

    y = actuals.values
    spike_mask = y > spike_threshold

    for model, df in forecasts.items():
        prefix = model  # e.g. "lgbm" or "lear"
        q05 = df[f"{prefix}_q05"].values
        q50 = df[f"{prefix}_q50"].values
        q95 = df[f"{prefix}_q95"].values

        mask = ~(np.isnan(y) | np.isnan(q50))
        if mask.sum() == 0:
            continue

        results["crps"][model] = _crps_quantile(
            y[mask], q05[mask], q50[mask], q95[mask]
        )
        results["mae"][model] = float(np.mean(np.abs(y[mask] - q50[mask])))
        results["coverage"][model] = float(
            np.mean((y[mask] >= q05[mask]) & (y[mask] <= q95[mask]))
        )
        results["interval_width"][model] = float(np.mean(q95[mask] - q05[mask]))

        spike_and_valid = mask & spike_mask
        if spike_and_valid.sum() > 0:
            results["spike_mae"][model] = float(
                np.mean(np.abs(y[spike_and_valid] - q50[spike_and_valid]))
            )

    return results


def _reliability_data(
    actuals: pd.Series,
    forecasts: dict[str, pd.DataFrame],
    quantile_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Compute observed coverage at each nominal quantile level.

    For a perfectly calibrated model, observed_coverage ~= nominal_level.
    """
    if quantile_levels is None:
        quantile_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    y = actuals.values
    rows = []

    for model, df in forecasts.items():
        prefix = model
        q50 = df[f"{prefix}_q50"].values
        q05 = df[f"{prefix}_q05"].values
        q95 = df[f"{prefix}_q95"].values
        mask = ~(np.isnan(y) | np.isnan(q50))

        # Fit a Gaussian from q05/q95 to evaluate arbitrary quantiles
        mu = q50[mask]
        sigma = np.clip((q95[mask] - q05[mask]) / 3.29, 1e-3, None)

        for ql in quantile_levels:
            from scipy.stats import norm

            threshold = norm.ppf(ql, loc=mu, scale=sigma)
            obs_coverage = float(np.mean(y[mask] <= threshold))
            rows.append({"model": model, "nominal": ql, "observed": obs_coverage})

    return pd.DataFrame(rows)


# -- Walk-forward engine -------------------------------------------------------


def walk_forward(
    td: TimeDBClient,
    train_model_fns: dict[str, callable],
    predict_fns: dict[str, callable],
    *,
    start: datetime,
    end: datetime,
    train_days: int = 365,
    test_days: int = 7,
    step_days: int = 7,
    spike_threshold: float = 100.0,
    verbose: bool = True,
) -> EvalResult:
    """Run rolling-window evaluation.

    Args:
        td:               Active TimeDBClient.
        train_model_fns:  Dict model_name -> callable(df) that trains in-place.
        predict_fns:      Dict model_name -> callable(df) -> forecast DataFrame.
        start:            First training window start.
        end:              Last test window end.
        train_days:       Rolling training window size.
        test_days:        Test window per fold.
        step_days:        Days to advance between folds.
        spike_threshold:  EUR/MWh threshold for spike MAE.
        verbose:          Print fold progress.

    Returns:
        EvalResult with per-fold results and aggregated summary.
    """
    fold_results = []
    fold = 0
    cursor = start + timedelta(days=train_days)

    while cursor + timedelta(days=test_days) <= end:
        train_start = cursor - timedelta(days=train_days)
        train_end = cursor
        test_start = cursor
        test_end = cursor + timedelta(days=test_days)

        # Feature matrices
        train_df = build_features(td, train_start, train_end)
        test_df = build_features(td, test_start, test_end)

        train_labelled = train_df.dropna(subset=["price"])
        test_labelled = test_df.dropna(subset=["price"])

        if len(train_labelled) < 24 or len(test_labelled) < 24:
            cursor += timedelta(days=step_days)
            fold += 1
            continue

        if verbose:
            print(
                f"Fold {fold+1}: train {train_start.date()} -> {train_end.date()} "
                f"| test {test_start.date()} -> {test_end.date()} "
                f"| n_train={len(train_labelled)}"
            )

        # Train all models on this fold's window
        for name, train_fn in train_model_fns.items():
            train_fn(train_labelled, verbose=False)

        # Predict on test window
        forecasts = {
            name: predict_fn(test_df) for name, predict_fn in predict_fns.items()
        }

        # Align actuals with forecast index
        actuals = test_df["price"].reindex(forecasts[next(iter(forecasts))].index)

        # Compute metrics
        metrics = _compute_metrics(actuals, forecasts, spike_threshold)
        fr = FoldResult(
            fold=fold,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            n_train=len(train_labelled),
            n_test=len(test_labelled),
            crps=metrics["crps"],
            mae=metrics["mae"],
            spike_mae=metrics["spike_mae"],
            coverage=metrics["coverage"],
            interval_width=metrics["interval_width"],
            actuals=actuals,
            preds=forecasts,
            reliability=_reliability_data(actuals, forecasts),
        )
        fold_results.append(fr)

        cursor += timedelta(days=step_days)
        fold += 1

    # -- Aggregate -------------------------------------------------------------
    summary_rows = []
    all_rel = []

    for fr in fold_results:
        for model in fr.crps:
            summary_rows.append(
                {
                    "fold": fr.fold,
                    "model": model,
                    "crps": fr.crps.get(model),
                    "mae": fr.mae.get(model),
                    "spike_mae": fr.spike_mae.get(model),
                    "coverage": fr.coverage.get(model),
                    "interval_width": fr.interval_width.get(model),
                }
            )
        if fr.preds:
            all_rel.append(_reliability_data(fr.actuals, fr.preds))

    summary_df = (
        pd.DataFrame(summary_rows)
        .groupby("model")[["crps", "mae", "spike_mae", "coverage", "interval_width"]]
        .agg(["mean", "std"])
    )
    summary_df.columns = ["_".join(c) for c in summary_df.columns]

    reliability_df = (
        pd.concat(all_rel)
        .groupby(["model", "nominal"])["observed"]
        .mean()
        .reset_index()
    )

    return EvalResult(
        folds=fold_results, summary=summary_df, reliability=reliability_df
    )


# -- CLI smoke-test ------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    from ml.models import lear, lgbm

    # Quick sanity-check on synthetic data (no DB needed)
    rng = np.random.default_rng(0)
    n = 24 * 400
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    from ml.models.lgbm import FEATURE_COLS

    fake = pd.DataFrame(
        {col: rng.standard_normal(n) for col in FEATURE_COLS}, index=idx
    )
    fake.index.name = "valid_time"
    fake["price"] = (
        50
        + 10 * fake["hour_sin"]
        + 5 * fake["temperature"]
        + rng.standard_normal(n) * 15
    )

    # Train once on full fake data
    print("Smoke-test: training on synthetic data ...")
    lgbm.train(fake, verbose=False)
    lear.train(fake, verbose=False)

    # Evaluate predictions on the same data (in-sample - just testing pipeline)
    actuals = fake["price"]
    forecasts = {
        "lgbm": lgbm.predict(fake),
        "lear": lear.predict(fake),
    }

    metrics = _compute_metrics(actuals, forecasts)
    print("\nIn-sample metrics (smoke-test only - not out-of-sample):")
    for metric, vals in metrics.items():
        for model, val in vals.items():
            print(f"  {model:6s}  {metric:15s}  {val:.3f}")

    rel = _reliability_data(actuals, forecasts)
    print("\nReliability (first 5 rows):")
    print(rel.head().to_string(index=False))
