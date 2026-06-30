"""Forecast quality metrics for NordSpot monitoring.

Three metrics are computed to detect model degradation:

1. rMAE (relative MAE)
   MAE of the ensemble q50 divided by the MAE of a naive same-hour-yesterday
   baseline.  Values > 1 mean the model is worse than the naive baseline.
   rMAE is robust to volatile market periods: when prices spike, both model
   MAE and naive MAE rise proportionally, so rMAE stays stable.  Only genuine
   model degradation causes rMAE to increase.

2. Pinball loss (q50)
   The exact training objective for the q50 quantile (equivalent to MAE / 2
   for symmetric quantile loss at q=0.5).  Tracked as an absolute signal;
   alerts when the 7-day rolling value exceeds 1.5x the long-run average.

3. Coverage rate
   Fraction of actual prices falling inside [ens_q05, ens_q95].  Calibrated
   for 90 % coverage via split-conformal correction.  Coverage below 80 %
   signals interval collapse; above 97 % signals the model is over-cautious
   (intervals too wide, less useful for trading decisions).

All functions operate on aligned pandas Series with a UTC DatetimeIndex.
Missing values (NaN) are dropped before computation so partial days do not
produce misleading aggregates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actuals: pd.Series, point_forecasts: pd.Series) -> float:
    """Mean absolute error between actuals and point forecasts (EUR/MWh).

    NaN rows are dropped before computation.

    Parameters
    ----------
    actuals:
        Observed prices (EUR/MWh).
    point_forecasts:
        Ensemble q50 predictions (EUR/MWh).

    Returns
    -------
    float
        MAE in EUR/MWh.  Returns NaN if fewer than 2 aligned non-NaN rows.
    """
    idx = actuals.dropna().index.intersection(point_forecasts.dropna().index)
    if len(idx) < 2:
        return float("nan")
    return float(np.mean(np.abs(actuals[idx].values - point_forecasts[idx].values)))


def naive_mae(actuals: pd.Series) -> float:
    """MAE of the same-hour-yesterday naive baseline.

    The naive forecast for hour h on day D is the actual price at hour h on
    day D-1.  This is the standard benchmark for day-ahead electricity price
    forecasting (Weron 2014, Lago et al. 2021).

    Parameters
    ----------
    actuals:
        Observed prices with a UTC hourly DatetimeIndex spanning at least
        25 hours (so at least one lag-24h value is available).

    Returns
    -------
    float
        Naive MAE in EUR/MWh.  Returns NaN if there are fewer than 2 valid pairs.
    """
    lag = actuals.shift(24)
    idx = actuals.dropna().index.intersection(lag.dropna().index)
    if len(idx) < 2:
        return float("nan")
    return float(np.mean(np.abs(actuals[idx].values - lag[idx].values)))


def relative_mae(actuals: pd.Series, point_forecasts: pd.Series) -> float:
    """Relative MAE: model MAE divided by naive same-hour-yesterday MAE.

    Values < 1 mean the model beats the naive baseline.
    Values > 1 mean the model is worse - a strong degradation signal.

    Parameters
    ----------
    actuals:
        Observed prices (EUR/MWh).
    point_forecasts:
        Ensemble q50 predictions (EUR/MWh).

    Returns
    -------
    float
        rMAE (dimensionless).  Returns NaN if either component is NaN or
        if the naive MAE is zero (degenerate series).
    """
    model = mae(actuals, point_forecasts)
    naive = naive_mae(actuals)
    if np.isnan(model) or np.isnan(naive) or naive == 0.0:
        return float("nan")
    return model / naive


def pinball_loss(
    actuals: pd.Series,
    quantile_forecasts: pd.Series,
    q: float,
) -> float:
    """Pinball loss (quantile loss) for a single quantile.

    For quantile q:
        L_q(y, y_hat) = q * (y - y_hat)       if y >= y_hat
                      = (1 - q) * (y_hat - y)  otherwise

    At q=0.5 this equals MAE / 2.

    Parameters
    ----------
    actuals:
        Observed prices (EUR/MWh).
    quantile_forecasts:
        Predicted quantile (EUR/MWh).
    q:
        Quantile level in (0, 1).

    Returns
    -------
    float
        Mean pinball loss in EUR/MWh.  Returns NaN if fewer than 2 valid rows.
    """
    idx = actuals.dropna().index.intersection(quantile_forecasts.dropna().index)
    if len(idx) < 2:
        return float("nan")
    y = actuals[idx].values
    y_hat = quantile_forecasts[idx].values
    errors = np.where(y >= y_hat, q * (y - y_hat), (1.0 - q) * (y_hat - y))
    return float(np.mean(errors))


def coverage_rate(
    actuals: pd.Series,
    q05_forecasts: pd.Series,
    q95_forecasts: pd.Series,
) -> float:
    """Fraction of actuals falling within [q05, q95].

    Calibrated target is 0.90.  Values outside [0.80, 0.97] trigger an alert.

    Parameters
    ----------
    actuals:
        Observed prices (EUR/MWh).
    q05_forecasts:
        Lower bound of the 90 % prediction interval.
    q95_forecasts:
        Upper bound of the 90 % prediction interval.

    Returns
    -------
    float
        Coverage rate in [0, 1].  Returns NaN if fewer than 2 valid rows.
    """
    idx = (
        actuals.dropna()
        .index.intersection(q05_forecasts.dropna().index)
        .intersection(q95_forecasts.dropna().index)
    )
    if len(idx) < 2:
        return float("nan")
    y = actuals[idx].values
    lo = q05_forecasts[idx].values
    hi = q95_forecasts[idx].values
    return float(np.mean((y >= lo) & (y <= hi)))


def compute_rolling_metrics(
    actuals: pd.Series,
    ens_q05: pd.Series,
    ens_q50: pd.Series,
    ens_q95: pd.Series,
    window_days: int = 7,
) -> dict[str, float]:
    """Compute all monitoring metrics over the most recent `window_days` days.

    Slices to the last `window_days * 24` hours before computing so daily
    monitoring runs always operate on the same window length regardless of
    how much history is stored.

    Parameters
    ----------
    actuals:
        Full history of observed prices.
    ens_q05, ens_q50, ens_q95:
        Full history of ensemble forecast quantiles.
    window_days:
        Rolling window length in days (default 7).

    Returns
    -------
    dict with keys:
        mae_eur          - absolute MAE of q50 (EUR/MWh)
        naive_mae_eur    - same-hour-yesterday MAE (EUR/MWh)
        rmae             - relative MAE (dimensionless)
        pinball_q50      - pinball loss at q=0.50 (EUR/MWh)
        pinball_q05      - pinball loss at q=0.05 (EUR/MWh)
        pinball_q95      - pinball loss at q=0.95 (EUR/MWh)
        coverage_rate    - fraction of actuals inside [q05, q95]
        n_hours          - number of aligned non-NaN rows in the window
    """
    # Slice to rolling window
    n_hours = window_days * 24
    act = actuals.iloc[-n_hours:] if len(actuals) > n_hours else actuals
    q05 = ens_q05.reindex(act.index)
    q50 = ens_q50.reindex(act.index)
    q95 = ens_q95.reindex(act.index)

    # Count aligned non-NaN rows for q50 (sanity check)
    valid_idx = act.dropna().index.intersection(q50.dropna().index)

    return {
        "mae_eur": mae(act, q50),
        "naive_mae_eur": naive_mae(act),
        "rmae": relative_mae(act, q50),
        "pinball_q50": pinball_loss(act, q50, q=0.50),
        "pinball_q05": pinball_loss(act, q05, q=0.05),
        "pinball_q95": pinball_loss(act, q95, q=0.95),
        "coverage_rate": coverage_rate(act, q05, q95),
        "n_hours": len(valid_idx),
    }
