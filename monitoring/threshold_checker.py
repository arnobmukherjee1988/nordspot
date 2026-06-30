"""Threshold-based alert checker for NordSpot forecast monitoring.

Reads per-zone thresholds from config/thresholds.yaml and compares them
against the rolling metrics produced by monitoring/metrics.py.

Usage
-----
    from monitoring.threshold_checker import check_thresholds

    alerts = check_thresholds(
        zone="SE3",
        metrics={"rmae": 1.35, "pinball_q50": 18.2, "coverage_rate": 0.75, ...},
        baseline_pinball_q50=12.1,
        n_hours=168,
    )
    # alerts -> [{"metric": "rmae", "value": 1.35, "threshold": 1.2, "reason": "..."}]
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

_THRESHOLDS_PATH = Path(__file__).parent.parent / "config" / "thresholds.yaml"


def load_thresholds(zone: str) -> dict[str, Any]:
    """Load per-zone thresholds from config/thresholds.yaml.

    Parameters
    ----------
    zone:
        Zone identifier (e.g. "SE3").

    Returns
    -------
    dict with keys matching the YAML zone block plus top-level window_days
    and baseline_days.

    Raises
    ------
    KeyError
        If the zone is not found in the thresholds file.
    """
    with open(_THRESHOLDS_PATH) as f:
        cfg = yaml.safe_load(f)

    if zone not in cfg.get("zones", {}):
        raise KeyError(
            f"Zone '{zone}' not found in {_THRESHOLDS_PATH}. "
            f"Available zones: {list(cfg.get('zones', {}).keys())}"
        )

    result = dict(cfg["zones"][zone])
    result["window_days"] = cfg.get("window_days", 7)
    result["baseline_days"] = cfg.get("baseline_days", 90)
    return result


def check_thresholds(
    zone: str,
    metrics: dict[str, float],
    baseline_pinball_q50: float | None = None,
) -> list[dict[str, Any]]:
    """Compare rolling metrics against per-zone thresholds.

    Parameters
    ----------
    zone:
        Zone identifier (e.g. "SE3").
    metrics:
        Output of monitoring.metrics.compute_rolling_metrics().
        Expected keys: rmae, pinball_q50, coverage_rate, n_hours.
    baseline_pinball_q50:
        Long-run (90-day) average pinball loss at q50 (EUR/MWh).
        Required for the pinball_q50_multiplier threshold check.
        If None, that check is skipped.

    Returns
    -------
    list of alert dicts.  Each dict has:
        metric     - metric name that breached
        value      - observed value
        threshold  - the threshold that was breached
        reason     - human-readable explanation
    Empty list means no alerts.
    """
    thresholds = load_thresholds(zone)
    alerts: list[dict[str, Any]] = []

    n_hours = metrics.get("n_hours", 0)
    min_hours = thresholds.get("min_hours", 48)

    # Not enough data yet - skip alerting
    if n_hours < min_hours:
        return []

    # -- rMAE check --------------------------------------------------------
    rmae = metrics.get("rmae", float("nan"))
    rmae_max = thresholds["rmae_max"]
    if not math.isnan(rmae) and rmae > rmae_max:
        alerts.append(
            {
                "metric": "rmae",
                "value": round(rmae, 4),
                "threshold": rmae_max,
                "reason": (
                    f"rMAE {rmae:.3f} exceeds {rmae_max:.2f}: model is "
                    f"{(rmae - 1) * 100:.1f}% worse than the naive baseline."
                ),
            }
        )

    # -- Pinball loss check ------------------------------------------------
    pinball = metrics.get("pinball_q50", float("nan"))
    multiplier = thresholds["pinball_q50_multiplier"]
    if (
        not math.isnan(pinball)
        and baseline_pinball_q50 is not None
        and not math.isnan(baseline_pinball_q50)
        and baseline_pinball_q50 > 0
    ):
        pinball_threshold = baseline_pinball_q50 * multiplier
        if pinball > pinball_threshold:
            alerts.append(
                {
                    "metric": "pinball_q50",
                    "value": round(pinball, 4),
                    "threshold": round(pinball_threshold, 4),
                    "reason": (
                        f"7-day pinball loss {pinball:.2f} EUR/MWh exceeds "
                        f"{multiplier}x the 90-day baseline "
                        f"({baseline_pinball_q50:.2f} EUR/MWh)."
                    ),
                }
            )

    # -- Coverage rate checks ----------------------------------------------
    cov = metrics.get("coverage_rate", float("nan"))
    cov_min = thresholds["coverage_min"]
    cov_max = thresholds["coverage_max"]
    if not math.isnan(cov):
        if cov < cov_min:
            alerts.append(
                {
                    "metric": "coverage_rate",
                    "value": round(cov, 4),
                    "threshold": cov_min,
                    "reason": (
                        f"Coverage {cov:.1%} is below minimum {cov_min:.0%}: "
                        "prediction intervals are too tight."
                    ),
                }
            )
        elif cov > cov_max:
            alerts.append(
                {
                    "metric": "coverage_rate",
                    "value": round(cov, 4),
                    "threshold": cov_max,
                    "reason": (
                        f"Coverage {cov:.1%} exceeds maximum {cov_max:.0%}: "
                        "prediction intervals are too wide."
                    ),
                }
            )

    return alerts
