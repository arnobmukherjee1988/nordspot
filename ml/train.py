"""Train all models on historical SE3 data stored in TimeDB.
Saves model/metrics.json and appends an entry to model/MODEL_LOG.md.

Usage:
    python -m ml.train
    python -m ml.train --start 2022-01-01
    python -m ml.train --note "Added 336h lag, recency weights"
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from timedb import TimeDBClient

from db.schema import init_schema, SERIES
from pipeline.features import build_features
from pipeline.store import write_series
from ml.models import lgbm, lear

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
MODEL_DIR.mkdir(exist_ok=True)


# ── Training cache ────────────────────────────────────────────────────────────

def _should_skip_training(td: TimeDBClient) -> bool:
    """Return True if models are fresh enough that retraining can be skipped.

    Criteria: trained_at.json exists, all model files exist, AND the gap
    between the last training time and the latest price row in TimeDB is
    less than 7 days (i.e. less than one week of new data has arrived).
    Stale models are worse than no models, so we err on the side of retraining.
    """
    trained_path = MODEL_DIR / "trained_at.json"
    model_files  = [MODEL_DIR / "lgbm_q50.pkl", MODEL_DIR / "lear_h00.pkl"]

    if not all(p.exists() for p in [trained_path, *model_files]):
        return False

    with open(trained_path) as f:
        info = json.load(f)

    trained_iso = info.get("trained_at", "")
    if not trained_iso:
        return False
    trained_dt = datetime.fromisoformat(trained_iso)
    if trained_dt.tzinfo is None:
        trained_dt = trained_dt.replace(tzinfo=timezone.utc)

    try:
        raw = td.read(series_ids=[SERIES["prices_raw"]], retention="forever")
        pdf = raw.to_pandas()
        if len(pdf) == 0:
            return False
        pdf["valid_time"] = pd.to_datetime(pdf["valid_time"], utc=True)
        latest_price_time = pdf["valid_time"].max()
    except Exception as exc:
        print(f"  [WARN] Cache check failed (TimeDB read error): {exc}")
        return False

    gap_days = (latest_price_time - trained_dt).total_seconds() / 86_400
    if gap_days < 7:
        print(
            f"\n[SKIP] Skipping retraining - models trained {trained_dt.date()}, "
            f"latest price data {latest_price_time.date()} "
            f"({gap_days:.1f} days gap, threshold = 7 days).\n"
            f"   Pass --force to retrain anyway.\n"
        )
        return True
    return False


def _write_forecasts_to_timedb(
    td: TimeDBClient,
    lgbm_val: pd.DataFrame,
    lear_val: pd.DataFrame,
    knowledge_time: datetime,
) -> None:
    """Persist holdout-period forecasts to TimeDB with the run's knowledge_time.

    Using knowledge_time lets you later ask: "what did the model predict for
    <valid_time> as of <knowledge_time>?" — the core bitemporal use-case.
    """
    mapping = [
        ("lgbm_q05", lgbm_val["lgbm_q05"]),
        ("lgbm_q50", lgbm_val["lgbm_q50"]),
        ("lgbm_q95", lgbm_val["lgbm_q95"]),
        ("lear_q05", lear_val["lear_q05"]),
        ("lear_q50", lear_val["lear_q50"]),
        ("lear_q95", lear_val["lear_q95"]),
    ]
    n_written = 0
    for name, series in mapping:
        s = series.dropna()
        if s.empty:
            continue
        df_out = pd.DataFrame({"valid_time": s.index, "value": s.values})
        write_series(td, SERIES[name], df_out, retention="forever",
                     knowledge_time=knowledge_time)
        n_written += len(df_out)

    lgbm_rows = lgbm_val["lgbm_q50"].notna().sum()
    lear_rows  = lear_val["lear_q50"].notna().sum()
    print(
        f"  [OK] Forecasts written to TimeDB  "
        f"(LGBM: {lgbm_rows} rows x 3 quantiles, "
        f"LEAR: {lear_rows} rows x 3 quantiles, "
        f"knowledge_time={knowledge_time.strftime('%Y-%m-%d %H:%M')} UTC)"
    )


# ── Metrics computation ───────────────────────────────────────────────────────

def _compute_metrics(
    actuals: pd.Series,
    q05: pd.Series,
    q50: pd.Series,
    q95: pd.Series,
    test_from: str,
    test_to: str,
) -> dict:
    mask    = actuals.notna() & q50.notna()
    y       = actuals[mask].values
    f50     = q50[mask].values
    f05     = q05[mask].values
    f95     = q95[mask].values
    hrs     = actuals[mask].index

    err     = f50 - y
    abs_err = np.abs(err)
    mae     = float(abs_err.mean())
    rmse    = float(np.sqrt((err ** 2).mean()))

    nonzero = np.abs(y) >= 1.0
    mape    = float(np.mean(np.abs(err[nonzero] / y[nonzero])) * 100) if nonzero.sum() else 0.0

    inside   = (y >= f05) & (y <= f95)
    coverage = float(inside.mean() * 100)

    spike_mask = y > 100
    spike_mae  = float(abs_err[spike_mask].mean()) if spike_mask.sum() else 0.0
    n_spikes   = int(spike_mask.sum())

    hour_series = pd.Series(abs_err, index=hrs)
    mae_by_hour = {
        str(h): float(v)
        for h, v in hour_series.groupby(hour_series.index.hour).mean().items()
    }

    night_hours = [0, 1, 2, 3, 4, 5, 23]
    peak_hours  = [8, 9, 17, 18, 19, 20]
    night_mask  = pd.Series(hrs.hour).isin(night_hours).values
    peak_mask   = pd.Series(hrs.hour).isin(peak_hours).values
    night_mae   = float(abs_err[night_mask].mean()) if night_mask.sum() else 0.0
    peak_mae    = float(abs_err[peak_mask].mean())  if peak_mask.sum()  else 0.0

    return {
        "test_from": test_from,
        "test_to":   test_to,
        "n_hours":   int(mask.sum()),
        "mae":       mae,
        "rmse":      rmse,
        "mape":      mape,
        "coverage_q5_q95": coverage,
        "spike_mae": spike_mae,
        "n_spikes":  n_spikes,
        "mae_by_hour": mae_by_hour,
        "night_mae": night_mae,
        "peak_mae":  peak_mae,
    }


# ── Model log ─────────────────────────────────────────────────────────────────

def _append_log(
    run_time: str,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    n_train: int,
    prev_metrics: dict | None,
    new_metrics: dict,
    note: str,
    lgbm_best_iters: list[int],
) -> None:
    log_path = MODEL_DIR / "MODEL_LOG.md"

    def _fmt(m: dict, model: str) -> str:
        mm = m.get(model, {})
        cov = mm.get("coverage_q5_q95", 0)
        cov_raw = mm.get("coverage_q5_q95_raw")
        cov_str = (
            f"{cov_raw:.1f}%->{cov:.1f}% [OK]" if cov_raw is not None else f"{cov:.1f}%"
        )
        return (
            f"| {model.upper():10s} "
            f"| {mm.get('mae', 0):.2f} "
            f"| {mm.get('rmse', 0):.2f} "
            f"| {cov_str} "
            f"| {mm.get('spike_mae', 0):.2f} "
            f"| {mm.get('night_mae', 0):.2f} "
            f"| {mm.get('peak_mae', 0):.2f} |"
        )

    header = "| Model | MAE | RMSE | Coverage | Spike MAE | Night MAE | Peak MAE |"
    sep    = "|---|---|---|---|---|---|---|"

    entry_lines = [
        f"\n---\n",
        f"## Run — {run_time} UTC\n",
        f"**Train:** {train_start} -> {train_end}  ({n_train:,} labelled rows)  ",
        f"**Holdout:** {val_start} -> {val_end}\n",
        f"**Note:** {note}\n",
    ]

    if prev_metrics:
        entry_lines += [
            "### Before\n",
            header, sep,
            _fmt(prev_metrics, "lgbm"),
            _fmt(prev_metrics, "lear"),
            "",
        ]

    entry_lines += [
        "### After\n",
        header, sep,
        _fmt(new_metrics, "lgbm"),
        _fmt(new_metrics, "lear"),
        "",
    ]

    if prev_metrics:
        lgbm_prev = prev_metrics.get("lgbm", {})
        lgbm_curr = new_metrics.get("lgbm", {})
        lear_prev = prev_metrics.get("lear", {})
        lear_curr = new_metrics.get("lear", {})
        d_lgbm_mae = lgbm_curr.get("mae", 0) - lgbm_prev.get("mae", 0)
        d_lear_mae = lear_curr.get("mae", 0) - lear_prev.get("mae", 0)
        d_lgbm_cov = lgbm_curr.get("coverage_q5_q95", 0) - lgbm_prev.get("coverage_q5_q95", 0)
        d_lear_cov = lear_curr.get("coverage_q5_q95", 0) - lear_prev.get("coverage_q5_q95", 0)

        def _delta(v: float, reverse: bool = False) -> str:
            sign = "+" if v > 0 else "-"
            good = (v < 0) if not reverse else (v > 0)
            icon = "[OK]" if good else "[!]"
            return f"{sign}{abs(v):.2f} {icon}"

        entry_lines += [
            "### Delta vs previous run\n",
            f"| | MAE delta | Coverage delta |",
            f"|---|---|---|",
            f"| LightGBM | {_delta(d_lgbm_mae)} | {_delta(d_lgbm_cov, reverse=True)} |",
            f"| LEAR | {_delta(d_lear_mae)} | {_delta(d_lear_cov, reverse=True)} |",
            "",
        ]

    if lgbm_best_iters:
        iters_str = ", ".join(str(i) for i in lgbm_best_iters)
        entry_lines.append(f"**LightGBM early stopping best iterations:** {iters_str}\n")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(entry_lines) + "\n")

    print(f"\n[LOG] Log appended -> {log_path}")


# ── Console results table ─────────────────────────────────────────────────────

def _print_metrics_table(
    title: str,
    lgbm_m: dict,
    lear_m: dict,
    lgbm_raw_cov: float | None = None,
) -> None:
    """Print a box-drawing results table matching the run_eval output style."""
    W = 78
    print(f"\n{'-' * W}")
    print(f"  {title}")
    print(f"{'-' * W}")
    print(
        f"  {'Model':8s}  {'MAE':>7s}  {'RMSE':>7s}  "
        f"{'Coverage':>14s}  {'Spike MAE':>10s}  {'Night MAE':>10s}  {'Peak MAE':>10s}"
    )
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}")

    lgbm_cov = (
        f"{lgbm_raw_cov:.1f}%->{lgbm_m['coverage_q5_q95']:.1f}%"
        if lgbm_raw_cov is not None
        else f"{lgbm_m['coverage_q5_q95']:.1f}%"
    )
    print(
        f"  {'lgbm':8s}  {lgbm_m['mae']:7.2f}  {lgbm_m['rmse']:7.2f}  "
        f"{lgbm_cov:>14s}  {lgbm_m['spike_mae']:10.2f}  "
        f"{lgbm_m['night_mae']:10.2f}  {lgbm_m['peak_mae']:10.2f}"
    )
    lear_cov = f"{lear_m['coverage_q5_q95']:.1f}%"
    print(
        f"  {'lear':8s}  {lear_m['mae']:7.2f}  {lear_m['rmse']:7.2f}  "
        f"{lear_cov:>14s}  {lear_m['spike_mae']:10.2f}  "
        f"{lear_m['night_mae']:10.2f}  {lear_m['peak_mae']:10.2f}"
    )
    print(f"{'-' * W}")


# ── Main training routine ─────────────────────────────────────────────────────

def train(start: datetime, end: datetime, note: str = "Routine training run") -> None:
    td = init_schema()

    val_start  = end - timedelta(days=90)
    train_end  = val_start

    print(f"Loading training features {start.date()} -> {train_end.date()} ...")
    train_df  = build_features(td, start, train_end)
    labelled  = int(train_df["price"].notna().sum())
    print(f"  {len(train_df):,} rows, {labelled:,} labelled ({labelled/len(train_df)*100:.1f} %)")

    if labelled < 24 * 30:
        raise RuntimeError(
            f"Only {labelled} labelled rows — need at least 720. Is ENTSO-E data synced?"
        )

    # Read previous metrics (for delta comparison in log)
    metrics_path = MODEL_DIR / "metrics.json"
    prev_metrics = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            prev_metrics = json.load(f)

    print("\nTraining LightGBM ...")
    lgbm_models = lgbm.train(train_df)
    lgbm_best_iters = [
        (m.best_iteration_ or m.n_estimators) for m in lgbm_models.values()
    ]

    print("\nTraining LEAR ...")
    lear.train(train_df)

    print(f"\nEvaluating on holdout {val_start.date()} -> {end.date()} ...")
    val_df   = build_features(td, val_start, end)
    # apply_conformal=False: get raw model output so calibration sees true
    # uncorrected intervals — otherwise the old ĉ is baked in and the new
    # calibration always computes ĉ ≈ 0, destroying the correction.
    lgbm_val = lgbm.predict(val_df, apply_conformal=False)
    lear_val = lear.predict(val_df)
    actuals  = val_df["price"]

    # ── Pre-calibration metrics (raw quantile outputs) ────────────────────────
    lgbm_m_raw = _compute_metrics(
        actuals,
        lgbm_val["lgbm_q05"], lgbm_val["lgbm_q50"], lgbm_val["lgbm_q95"],
        test_from=str(val_start.date()), test_to=str(end.date()),
    )
    lear_m = _compute_metrics(
        actuals,
        lear_val["lear_q05"], lear_val["lear_q50"], lear_val["lear_q95"],
        test_from=str(val_start.date()), test_to=str(end.date()),
    )

    # ── Conformal calibration (LightGBM only) ────────────────────────────────
    # Fits the interval correction ĉ on the holdout so that future predictions
    # achieve TARGET_COVERAGE marginal coverage.  Saves lgbm_conformal.pkl.
    print("\nCalibrating LightGBM prediction intervals (split conformal) ...")
    mask = actuals.notna()
    lgbm.calibrate(
        actuals[mask],
        lgbm_val["lgbm_q05"][mask],
        lgbm_val["lgbm_q95"][mask],
    )

    # Re-predict on holdout with calibration applied to get final metrics
    lgbm_val_cal = lgbm.predict(val_df)
    lgbm_m = _compute_metrics(
        actuals,
        lgbm_val_cal["lgbm_q05"], lgbm_val_cal["lgbm_q50"], lgbm_val_cal["lgbm_q95"],
        test_from=str(val_start.date()), test_to=str(end.date()),
    )
    # Store raw coverage alongside calibrated for reference
    lgbm_m["coverage_q5_q95_raw"] = lgbm_m_raw["coverage_q5_q95"]

    new_metrics = {"lgbm": lgbm_m, "lear": lear_m}
    with open(metrics_path, "w") as f:
        json.dump(new_metrics, f, indent=2)

    _print_metrics_table(
        title=f"Holdout metrics -- {val_start.date()} -> {end.date()}",
        lgbm_m=lgbm_m,
        lear_m=lear_m,
        lgbm_raw_cov=lgbm_m_raw["coverage_q5_q95"],
    )

    run_kt   = datetime.now(timezone.utc)
    run_time = run_kt.strftime("%Y-%m-%d %H:%M")
    _append_log(
        run_time=run_time,
        train_start=str(start.date()),
        train_end=str(train_end.date()),
        val_start=str(val_start.date()),
        val_end=str(end.date()),
        n_train=labelled,
        prev_metrics=prev_metrics,
        new_metrics=new_metrics,
        note=note,
        lgbm_best_iters=lgbm_best_iters,
    )

    print("\nWriting forecasts to TimeDB ...")
    _write_forecasts_to_timedb(td, lgbm_val, lear_val, knowledge_time=run_kt)

    trained_info = {
        "trained_at":  run_kt.isoformat(),
        "train_end":   str(train_end.date()),
        "val_start":   str(val_start.date()),
        "val_end":     str(end.date()),
    }
    trained_path = MODEL_DIR / "trained_at.json"
    with open(trained_path, "w") as f:
        json.dump(trained_info, f, indent=2)
    print(f"  [OK] Training cache written -> {trained_path}")

    print(f"\n[OK] All models trained. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        default=os.getenv("SE3_TRAIN_START", "2020-01-01"),
        help="Training start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--note",
        default="Routine training run",
        help="Short description of changes made (logged to MODEL_LOG.md)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if the model cache is fresh (< 7 days old).",
    )
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    if not args.force:
        _td = init_schema()
        if _should_skip_training(_td):
            import sys
            sys.exit(0)

    train(start, end, note=args.note)
