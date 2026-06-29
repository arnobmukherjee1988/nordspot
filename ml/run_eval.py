"""Run evaluation on real SE3 data and append results to MODEL_LOG.md.

Two modes
---------
Quick mode (default)
    Uses already-trained models. Predicts on the 90-day holdout recorded in
    model/trained_at.json.  Completes in seconds; no retraining.

    python -m ml.run_eval

Walk-forward mode
    Retrains both models on a rolling window for every fold.  Produces
    statistically rigorous out-of-sample CRPS / MAE estimates but takes
    several hours depending on fold count.

    python -m ml.run_eval --walk-forward
    python -m ml.run_eval --walk-forward --train-days 180 --test-days 30 --step-days 60

Options
-------
    --walk-forward          Enable rolling re-training evaluation.
    --start YYYY-MM-DD      Start of the evaluation window (walk-forward only).
    --train-days N          Training window per fold (default 365).
    --test-days N           Test window per fold    (default 30).
    --step-days N           Step between folds      (default 90).
    --no-log                Print results but do not append to MODEL_LOG.md.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from db.schema import init_schema
from ml.evaluate import _compute_metrics, _crps_quantile, walk_forward
from ml.models import lear, lgbm
from pipeline.features import build_features

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))


# -- Helpers -------------------------------------------------------------------


def _print_table(title: str, rows: list[dict]) -> None:
    """Print a compact results table to stdout."""
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}")
    header = f"  {'Model':8s}  {'CRPS':>7s}  {'MAE':>7s}  {'Coverage':>10s}  {'Spike MAE':>10s}  {'Int Width':>10s}"
    print(header)
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in rows:
        print(
            f"  {r['model']:8s}  {r['crps']:7.3f}  {r['mae']:7.2f}  "
            f"{r['coverage']:9.1%}  {r['spike_mae']:10.2f}  {r['interval_width']:10.2f}"
        )
    print(f"{'-' * 70}")


def _append_eval_log(rows: list[dict], mode: str, period: str, note: str) -> None:
    log_path = MODEL_DIR / "MODEL_LOG.md"
    run_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    lines = [
        "\n---\n",
        f"## Evaluation Run - {run_time} UTC\n",
        f"**Mode:** {mode}  **Period:** {period}\n",
        f"**Note:** {note}\n",
        "| Model | CRPS | MAE | Coverage | Spike MAE | Interval Width |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model'].upper()} "
            f"| {r['crps']:.3f} "
            f"| {r['mae']:.2f} "
            f"| {r['coverage']:.1%} "
            f"| {r['spike_mae']:.2f} "
            f"| {r['interval_width']:.2f} |"
        )
    lines.append("")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[LOG] Results appended -> {log_path}")


# -- Quick evaluation (no retraining) -----------------------------------------


def run_quick(log: bool = True) -> None:
    """Evaluate already-trained models on the recorded holdout window."""
    trained_path = MODEL_DIR / "trained_at.json"
    if not trained_path.exists():
        raise FileNotFoundError(
            "model/trained_at.json not found - run 'python -m ml.train' first."
        )
    with open(trained_path) as f:
        info = json.load(f)

    val_start = datetime.fromisoformat(info["val_start"]).replace(tzinfo=timezone.utc)
    val_end = datetime.fromisoformat(info["val_end"]).replace(tzinfo=timezone.utc)

    print(f"Quick evaluation: holdout {val_start.date()} -> {val_end.date()}")
    print("  (Loading features from TimeDB ...)")

    td = init_schema()
    val_df = build_features(td, val_start, val_end)
    actuals = val_df["price"]

    lgbm_val = lgbm.predict(val_df)  # conformal correction applied automatically
    lear_val = lear.predict(val_df)

    forecasts = {"lgbm": lgbm_val, "lear": lear_val}
    metrics = _compute_metrics(actuals, forecasts)

    rows = []
    for model in ("lgbm", "lear"):
        # CRPS (Gaussian approximation from quantile spread)
        prefix = model
        fc = forecasts[model]
        mask = actuals.notna() & fc[f"{prefix}_q50"].notna()
        crps = _crps_quantile(
            actuals[mask].values,
            fc[f"{prefix}_q05"][mask].values,
            fc[f"{prefix}_q50"][mask].values,
            fc[f"{prefix}_q95"][mask].values,
        )
        rows.append(
            {
                "model": model,
                "crps": crps,
                "mae": metrics["mae"].get(model, float("nan")),
                "coverage": metrics["coverage"].get(model, float("nan")),
                "spike_mae": metrics["spike_mae"].get(model, float("nan")),
                "interval_width": metrics["interval_width"].get(model, float("nan")),
            }
        )

    period = f"{val_start.date()} -> {val_end.date()}"
    _print_table(f"Quick evaluation -- {period}", rows)

    if log:
        _append_eval_log(
            rows,
            mode="quick (no retraining)",
            period=period,
            note="Existing trained models; conformal correction applied to LGBM.",
        )


# -- Walk-forward evaluation ---------------------------------------------------


def run_walk_forward(
    start_iso: str,
    train_days: int,
    test_days: int,
    step_days: int,
    log: bool = True,
) -> None:
    """Rolling retrain + evaluate.  Takes O(hours) for many folds."""
    td = init_schema()
    start = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    n_folds_est = max(
        0, int((end - start - timedelta(days=train_days + test_days)).days / step_days)
    )
    print(
        f"\nWalk-forward: {start.date()} -> {end.date()} "
        f"| train={train_days}d  test={test_days}d  step={step_days}d "
        f"| ~{n_folds_est} folds"
    )
    print("  [WARN] This retrains both models on every fold - may take hours.\n")

    result = walk_forward(
        td=td,
        train_model_fns={"lgbm": lgbm.train, "lear": lear.train},
        predict_fns={"lgbm": lgbm.predict, "lear": lear.predict},
        start=start,
        end=end,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        verbose=True,
    )

    print("\nSummary across folds:")
    print(result.summary.round(3).to_string())

    rows = []
    for model in ("lgbm", "lear"):
        if model not in result.summary.index:
            continue
        s = result.summary.loc[model]
        rows.append(
            {
                "model": model,
                "crps": s.get("crps_mean", float("nan")),
                "mae": s.get("mae_mean", float("nan")),
                "coverage": s.get("coverage_mean", float("nan")),
                "spike_mae": s.get("spike_mae_mean", float("nan")),
                "interval_width": s.get("interval_width_mean", float("nan")),
            }
        )

    period = f"{start.date()} -> {end.date()}"
    n_folds = len(result.folds)
    _print_table(f"Walk-forward ({n_folds} folds) -- {period}", rows)

    if log:
        _append_eval_log(
            rows,
            mode=f"walk-forward ({n_folds} folds, train={train_days}d, step={step_days}d)",
            period=period,
            note="Rolling retrain; metrics are out-of-sample. CRPS = Gaussian approx from q05/q95 spread.",
        )


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Enable rolling retrain mode (slow).",
    )
    parser.add_argument(
        "--start",
        default="2022-01-01",
        help="Walk-forward start date (YYYY-MM-DD). Ignored in quick mode.",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=365,
        help="Training window in days (walk-forward only, default 365).",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Test window in days per fold (default 30).",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=90,
        help="Step between folds in days (default 90).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Print results but do not write to MODEL_LOG.md.",
    )
    args = parser.parse_args()

    if args.walk_forward:
        run_walk_forward(
            start_iso=args.start,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            log=not args.no_log,
        )
    else:
        run_quick(log=not args.no_log)
