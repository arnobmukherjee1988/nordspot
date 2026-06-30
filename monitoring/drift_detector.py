"""Daily drift detector for NordSpot forecast monitoring.

Reads stored predictions (ens_q05/q50/q95) and actual prices from TimeDB,
computes rolling quality metrics, checks against per-zone thresholds, and
outputs a JSON report.

Designed to run daily at 08:00 UTC (after the previous day's actual prices
have been published and ingested) via a cron job, Airflow DAG, or Makefile.

Usage
-----
    python -m monitoring.drift_detector --zone SE3
    python -m monitoring.drift_detector --zone SE3 --window 14 --output report.json
    python -m monitoring.drift_detector --zone ALL

Exit codes
----------
    0  - No alerts
    1  - One or more alert thresholds breached (use in CI/cron to trigger
         downstream retraining)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nordspot.monitoring.drift_detector")


def _read_series_to_series(td, series_id: int) -> pd.Series:
    """Pull a TimeDB series and return as a UTC-indexed pandas Series."""
    df = td.read(series_ids=[series_id], retention="forever")
    if len(df) == 0:
        return pd.Series(dtype=float)
    pdf = df.to_pandas()
    pdf["valid_time"] = pd.to_datetime(pdf["valid_time"], utc=True)
    return pdf.set_index("valid_time")["value"].sort_index()


def run_detector(
    zone: str,
    window_days: int = 7,
    baseline_days: int = 90,
    output_path: str | None = None,
) -> dict:
    """Run the full monitoring pipeline for a single zone.

    Parameters
    ----------
    zone:
        Zone identifier (e.g. "SE3").
    window_days:
        Rolling window for metric computation (default 7 days).
    baseline_days:
        Long-run baseline window for pinball threshold comparison (default 90).
    output_path:
        If set, write the JSON report to this file path in addition to stdout.

    Returns
    -------
    Report dict with keys:
        zone, run_at, window_days, metrics, alerts, drift_detected
    """
    from db.schema import SERIES, init_schema
    from monitoring.metrics import compute_rolling_metrics, pinball_loss
    from monitoring.threshold_checker import check_thresholds

    logger.info("Starting drift detection for zone %s (window=%dd)", zone, window_days)

    td = init_schema()

    # -- Load time series from TimeDB --------------------------------------
    actuals = _read_series_to_series(td, SERIES["prices_raw"])
    ens_q05 = _read_series_to_series(td, SERIES["ens_q05"])
    ens_q50 = _read_series_to_series(td, SERIES["ens_q50"])
    ens_q95 = _read_series_to_series(td, SERIES["ens_q95"])

    logger.info(
        "Loaded  actuals=%d rows  ens_q50=%d rows",
        len(actuals),
        len(ens_q50),
    )

    if ens_q50.empty:
        logger.warning(
            "No ensemble predictions found in TimeDB for zone %s. "
            "Has the API served at least one forecast request?",
            zone,
        )
        report = {
            "zone": zone,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "window_days": window_days,
            "metrics": {},
            "alerts": [],
            "drift_detected": False,
            "warning": "No forecast predictions in TimeDB yet.",
        }
        _output_report(report, output_path)
        return report

    # -- 7-day rolling metrics ---------------------------------------------
    metrics = compute_rolling_metrics(
        actuals=actuals,
        ens_q05=ens_q05,
        ens_q50=ens_q50,
        ens_q95=ens_q95,
        window_days=window_days,
    )

    # -- 90-day baseline pinball for relative threshold --------------------
    n_baseline = baseline_days * 24
    act_baseline = actuals.iloc[-n_baseline:] if len(actuals) > n_baseline else actuals
    q50_baseline = ens_q50.reindex(act_baseline.index)
    baseline_pinball = pinball_loss(act_baseline, q50_baseline, q=0.50)

    logger.info(
        "7-day metrics: MAE=%.2f  rMAE=%.3f  pinball_q50=%.2f  coverage=%.1f%%  n=%d",
        metrics.get("mae_eur", float("nan")),
        metrics.get("rmae", float("nan")),
        metrics.get("pinball_q50", float("nan")),
        metrics.get("coverage_rate", float("nan")) * 100,
        metrics.get("n_hours", 0),
    )

    # -- Check thresholds --------------------------------------------------
    alerts = check_thresholds(
        zone=zone,
        metrics=metrics,
        baseline_pinball_q50=baseline_pinball,
    )

    drift_detected = len(alerts) > 0

    # -- Build report ------------------------------------------------------
    report = {
        "zone": zone,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "baseline_days": baseline_days,
        "metrics": {
            k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
        },
        "baseline_pinball_q50": round(baseline_pinball, 4)
        if not pd.isna(baseline_pinball)
        else None,
        "alerts": alerts,
        "drift_detected": drift_detected,
    }

    if drift_detected:
        logger.warning(
            "DRIFT DETECTED for zone %s: %d alert(s) triggered",
            zone,
            len(alerts),
        )
        for a in alerts:
            logger.warning("  [ALERT] %s", a["reason"])
    else:
        logger.info(
            "No drift detected for zone %s - all metrics within thresholds.", zone
        )

    _output_report(report, output_path)
    return report


def _output_report(report: dict, output_path: str | None) -> None:
    """Print JSON report to stdout and optionally write to file."""
    report_json = json.dumps(report, indent=2, default=str)
    print(report_json)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report_json)
        logger.info("Report written to %s", output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NordSpot daily drift detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--zone",
        default="SE3",
        help="Zone to monitor (e.g. SE3) or ALL for all configured zones",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Rolling window in days (default 7)",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=90,
        help="Baseline window for pinball comparison in days (default 90)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON report file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.zone == "ALL":
        from config.zone_config import load_all_zones

        zones = list(load_all_zones().keys())
    else:
        zones = [args.zone]

    any_drift = False
    for zone in zones:
        report = run_detector(
            zone=zone,
            window_days=args.window,
            baseline_days=args.baseline,
            output_path=args.output,
        )
        if report.get("drift_detected"):
            any_drift = True

    sys.exit(1 if any_drift else 0)
