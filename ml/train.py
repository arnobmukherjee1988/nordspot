"""Train all models on historical SE3 data stored in TimeDB.
Saves model/metrics.json and appends an entry to model/MODEL_LOG.md.
Every run is tracked in MLflow (experiment: nordspot-lgbm).

Usage:
    python -m ml.train
    python -m ml.train --start 2022-01-01
    python -m ml.train --note "Added 336h lag, recency weights"
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import mlflow.catboost as mlflow_cat
import mlflow.lightgbm as mlflow_lgbm
import mlflow.sklearn as mlflow_sklearn
import mlflow.xgboost as mlflow_xgb
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from timedb import TimeDBClient

from config.zone_config import load_all_zones, load_zone
from db.schema import SERIES, init_schema
from ml.explain import log_shap_artifacts
from ml.mlflow_setup import EXPERIMENTS, get_tracking_uri
from ml.models import catboost as cat_model
from ml.models import ensemble as ens_model
from ml.models import lear, lgbm
from ml.models import xgboost as xgb_model
from ml.registry import register_and_promote
from pipeline.features import build_features
from pipeline.store import write_series

# -- S3 artifact sync ----------------------------------------------------------


def _s3_upload_models() -> None:
    """Upload model/ directory contents to S3 after training.

    Requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION,
    and S3_BUCKET in the environment. Silently skips if boto3 is not installed
    or S3_BUCKET is not set - so local runs without S3 still work.
    """
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return
    try:
        import boto3  # noqa: PLC0415
    except ImportError:
        print("  [WARN] boto3 not installed - skipping S3 upload")
        return

    s3 = boto3.client("s3")
    uploaded = 0
    for path in MODEL_DIR.rglob("*"):
        if path.is_file():
            key = f"model/{path.relative_to(MODEL_DIR)}"
            s3.upload_file(str(path), bucket, key)
            uploaded += 1
    print(f"  [OK] Uploaded {uploaded} model files to s3://{bucket}/model/")


MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
MODEL_DIR.mkdir(exist_ok=True)


# -- Training cache ------------------------------------------------------------


def _should_skip_training(td: TimeDBClient) -> bool:
    """Return True if models are fresh enough that retraining can be skipped.

    Criteria: trained_at.json exists, all model files exist, AND the gap
    between the last training time and the latest price row in TimeDB is
    less than 7 days (i.e. less than one week of new data has arrived).
    Stale models are worse than no models, so we err on the side of retraining.
    """
    trained_path = MODEL_DIR / "trained_at.json"
    model_files = [MODEL_DIR / "lgbm_q50.pkl", MODEL_DIR / "lear_h00.pkl"]

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
    <valid_time> as of <knowledge_time>?" - the core bitemporal use-case.
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
        write_series(
            td, SERIES[name], df_out, retention="forever", knowledge_time=knowledge_time
        )
        n_written += len(df_out)

    lgbm_rows = lgbm_val["lgbm_q50"].notna().sum()
    lear_rows = lear_val["lear_q50"].notna().sum()
    print(
        f"  [OK] Forecasts written to TimeDB  "
        f"(LGBM: {lgbm_rows} rows x 3 quantiles, "
        f"LEAR: {lear_rows} rows x 3 quantiles, "
        f"knowledge_time={knowledge_time.strftime('%Y-%m-%d %H:%M')} UTC)"
    )


# -- Metrics computation -------------------------------------------------------


def _compute_metrics(
    actuals: pd.Series,
    q05: pd.Series,
    q50: pd.Series,
    q95: pd.Series,
    test_from: str,
    test_to: str,
) -> dict:
    mask = actuals.notna() & q50.notna()
    y = actuals[mask].values
    f50 = q50[mask].values
    f05 = q05[mask].values
    f95 = q95[mask].values
    hrs = actuals[mask].index

    err = f50 - y
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err**2).mean()))

    nonzero = np.abs(y) >= 1.0
    mape = (
        float(np.mean(np.abs(err[nonzero] / y[nonzero])) * 100)
        if nonzero.sum()
        else 0.0
    )

    inside = (y >= f05) & (y <= f95)
    coverage = float(inside.mean() * 100)

    spike_mask = y > 100
    spike_mae = float(abs_err[spike_mask].mean()) if spike_mask.sum() else 0.0
    n_spikes = int(spike_mask.sum())

    hour_series = pd.Series(abs_err, index=hrs)
    mae_by_hour = {
        str(h): float(v)
        for h, v in hour_series.groupby(hour_series.index.hour).mean().items()
    }

    night_hours = [0, 1, 2, 3, 4, 5, 23]
    peak_hours = [8, 9, 17, 18, 19, 20]
    night_mask = pd.Series(hrs.hour).isin(night_hours).values
    peak_mask = pd.Series(hrs.hour).isin(peak_hours).values
    night_mae = float(abs_err[night_mask].mean()) if night_mask.sum() else 0.0
    peak_mae = float(abs_err[peak_mask].mean()) if peak_mask.sum() else 0.0

    return {
        "test_from": test_from,
        "test_to": test_to,
        "n_hours": int(mask.sum()),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "coverage_q5_q95": coverage,
        "spike_mae": spike_mae,
        "n_spikes": n_spikes,
        "mae_by_hour": mae_by_hour,
        "night_mae": night_mae,
        "peak_mae": peak_mae,
    }


# -- Model log -----------------------------------------------------------------


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
    sep = "|---|---|---|---|---|---|---|"

    entry_lines = [
        "\n---\n",
        f"## Run - {run_time} UTC\n",
        f"**Train:** {train_start} -> {train_end}  ({n_train:,} labelled rows)  ",
        f"**Holdout:** {val_start} -> {val_end}\n",
        f"**Note:** {note}\n",
    ]

    if prev_metrics:
        entry_lines += [
            "### Before\n",
            header,
            sep,
            _fmt(prev_metrics, "lgbm"),
            _fmt(prev_metrics, "lear"),
            "",
        ]

    entry_lines += [
        "### After\n",
        header,
        sep,
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
        d_lgbm_cov = lgbm_curr.get("coverage_q5_q95", 0) - lgbm_prev.get(
            "coverage_q5_q95", 0
        )
        d_lear_cov = lear_curr.get("coverage_q5_q95", 0) - lear_prev.get(
            "coverage_q5_q95", 0
        )

        def _delta(v: float, reverse: bool = False) -> str:
            sign = "+" if v > 0 else "-"
            good = (v < 0) if not reverse else (v > 0)
            icon = "[OK]" if good else "[!]"
            return f"{sign}{abs(v):.2f} {icon}"

        entry_lines += [
            "### Delta vs previous run\n",
            "| | MAE delta | Coverage delta |",
            "|---|---|---|",
            f"| LightGBM | {_delta(d_lgbm_mae)} | {_delta(d_lgbm_cov, reverse=True)} |",
            f"| LEAR | {_delta(d_lear_mae)} | {_delta(d_lear_cov, reverse=True)} |",
            "",
        ]

    if lgbm_best_iters:
        iters_str = ", ".join(str(i) for i in lgbm_best_iters)
        entry_lines.append(
            f"**LightGBM early stopping best iterations:** {iters_str}\n"
        )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(entry_lines) + "\n")

    print(f"\n[LOG] Log appended -> {log_path}")


# -- Console results table -----------------------------------------------------


def _print_metrics_table(
    title: str,
    lgbm_m: dict,
    lear_m: dict,
    lgbm_raw_cov: float | None = None,
) -> None:
    """Print a box-drawing results table matching the run_eval output style."""
    width = 78
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")
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
    print(f"{'-' * width}")


# -- MLflow: feature importance plot -------------------------------------------


def _log_feature_importance() -> None:
    """Log LightGBM feature importance bar chart to the active MLflow run.

    Silently skips if matplotlib is not installed or no trained models exist
    (e.g. unit tests that mock lgbm.train).
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fi = lgbm.feature_importance(top_n=20)
        fig, ax = plt.subplots(figsize=(8, 6))
        fi["mean"].sort_values().plot(kind="barh", ax=ax)
        ax.set_title("LightGBM Feature Importance (gain, mean across quantiles)")
        ax.set_xlabel("Importance (gain)")
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] Feature importance plot skipped: {exc}")


# -- Main training routine -----------------------------------------------------


def train(
    start: datetime,
    end: datetime,
    note: str = "Routine training run",
    zone: str = "SE3",
) -> None:
    # Validate and resolve zone config early - raises ValueError on unknown zone
    zone_cfg = load_zone(zone)

    # -- MLflow: set experiment ------------------------------------------------
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXPERIMENTS["lgbm"])

    with mlflow.start_run(run_name=f"lgbm-{end.date()}"):
        mlflow.set_tags(
            {
                "note": note,
                "zone": zone,
                "train_start": str(start.date()),
                "run_end": str(end.date()),
            }
        )

        td = init_schema()

        # -- Time splits -------------------------------------------------------
        # Timeline:
        #   |---- training data ----|---- 60d calibration ----|---- 30d test ----|
        #   start              cal_start                  test_start             end
        cal_start = end - timedelta(days=90)
        test_start = end - timedelta(days=30)
        train_end = cal_start

        print(f"Loading training features {start.date()} -> {train_end.date()} ...")
        train_df = build_features(zone_cfg, start, train_end, td=td)
        if "valid_time" in train_df.columns:
            train_df = train_df.set_index("valid_time")
        labelled = int(train_df["price"].notna().sum())
        print(
            f"  {len(train_df):,} rows, {labelled:,} labelled "
            f"({labelled / len(train_df) * 100:.1f} %)"
        )

        if labelled < 24 * 30:
            raise RuntimeError(
                f"Only {labelled} labelled rows - need at least 720. "
                "Is ENTSO-E data synced?"
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

        # -- Load calibration and test windows ---------------------------------
        print(
            f"\nLoading calibration window {cal_start.date()} -> {test_start.date()} ..."
        )
        cal_df = build_features(zone_cfg, cal_start, test_start, td=td)
        if "valid_time" in cal_df.columns:
            cal_df = cal_df.set_index("valid_time")

        print(f"Loading test window {test_start.date()} -> {end.date()} ...")
        test_df = build_features(zone_cfg, test_start, end, td=td)
        if "valid_time" in test_df.columns:
            test_df = test_df.set_index("valid_time")

        # -- Conformal calibration (LightGBM only) - on cal window ONLY --------
        print("\nCalibrating LightGBM prediction intervals (split conformal) ...")
        lgbm_cal_raw = lgbm.predict(cal_df, apply_conformal=False)
        cal_actuals = cal_df["price"]
        cal_mask = cal_actuals.notna()
        lgbm.calibrate(
            cal_actuals[cal_mask],
            lgbm_cal_raw["lgbm_q05"][cal_mask],
            lgbm_cal_raw["lgbm_q95"][cal_mask],
        )

        # -- Metrics on TEST window ---------------------------------------------
        print(f"\nEvaluating on test window {test_start.date()} -> {end.date()} ...")
        lgbm_test_raw = lgbm.predict(test_df, apply_conformal=False)
        lear_test = lear.predict(test_df)
        test_actuals = test_df["price"]

        lgbm_m_raw = _compute_metrics(
            test_actuals,
            lgbm_test_raw["lgbm_q05"],
            lgbm_test_raw["lgbm_q50"],
            lgbm_test_raw["lgbm_q95"],
            test_from=str(test_start.date()),
            test_to=str(end.date()),
        )

        lgbm_test_cal = lgbm.predict(test_df)
        lgbm_m = _compute_metrics(
            test_actuals,
            lgbm_test_cal["lgbm_q05"],
            lgbm_test_cal["lgbm_q50"],
            lgbm_test_cal["lgbm_q95"],
            test_from=str(test_start.date()),
            test_to=str(end.date()),
        )
        lgbm_m["coverage_q5_q95_raw"] = lgbm_m_raw["coverage_q5_q95"]

        lear_m = _compute_metrics(
            test_actuals,
            lear_test["lear_q05"],
            lear_test["lear_q50"],
            lear_test["lear_q95"],
            test_from=str(test_start.date()),
            test_to=str(end.date()),
        )

        new_metrics = {"lgbm": lgbm_m, "lear": lear_m}
        with open(metrics_path, "w") as f:
            json.dump(new_metrics, f, indent=2)

        _print_metrics_table(
            title=(
                f"Test metrics -- {test_start.date()} -> {end.date()}  "
                f"(calibrated on {cal_start.date()} -> {test_start.date()})"
            ),
            lgbm_m=lgbm_m,
            lear_m=lear_m,
            lgbm_raw_cov=lgbm_m_raw["coverage_q5_q95"],
        )

        run_kt = datetime.now(timezone.utc)
        run_time = run_kt.strftime("%Y-%m-%d %H:%M")
        _append_log(
            run_time=run_time,
            train_start=str(start.date()),
            train_end=str(train_end.date()),
            val_start=str(cal_start.date()),
            val_end=str(end.date()),
            n_train=labelled,
            prev_metrics=prev_metrics,
            new_metrics=new_metrics,
            note=note,
            lgbm_best_iters=lgbm_best_iters,
        )

        # -- MLflow: log hyperparameters ---------------------------------------
        _params_to_log = {
            k: v
            for k, v in lgbm._LGB_PARAMS_BASE.items()
            if k not in ("objective", "metric", "n_jobs", "verbose")
        }
        _params_to_log["lgbm_val_frac"] = lgbm.VAL_FRAC
        _params_to_log["lgbm_early_stop_rounds"] = lgbm.EARLY_STOP_N
        mlflow.log_params(_params_to_log)

        # -- MLflow: log metrics -----------------------------------------------
        mlflow.log_metrics(
            {
                "lgbm_mae": lgbm_m["mae"],
                "lgbm_rmse": lgbm_m["rmse"],
                "lgbm_mape": lgbm_m["mape"],
                "lgbm_coverage": lgbm_m["coverage_q5_q95"],
                "lgbm_spike_mae": lgbm_m["spike_mae"],
                "lgbm_night_mae": lgbm_m["night_mae"],
                "lgbm_peak_mae": lgbm_m["peak_mae"],
                "lear_mae": lear_m["mae"],
                "lear_rmse": lear_m["rmse"],
                "lear_coverage": lear_m["coverage_q5_q95"],
                "lear_spike_mae": lear_m["spike_mae"],
                "lear_night_mae": lear_m["night_mae"],
                "lear_peak_mae": lear_m["peak_mae"],
            }
        )

        # -- MLflow: log trained model artifacts (best-effort) -----------------
        # Artifact upload can fail when the MLflow server is local/Docker and
        # the artifact root is not reachable from the training host. Models are
        # already saved to model/ so this is supplementary.
        try:
            for name, model in lgbm_models.items():
                mlflow_lgbm.log_model(model, f"lgbm_{name}")
        except Exception as e:
            print(f"  [WARN] MLflow artifact upload skipped: {e}")

        # -- MLflow: log feature importance plot (best-effort) -----------------
        try:
            _log_feature_importance()
        except Exception as e:
            print(f"  [WARN] MLflow feature importance upload skipped: {e}")

        # -- MLflow: log SHAP interpretability plots (best-effort) -------------
        try:
            _lgbm_x, _ = lgbm._prep(
                train_df.sample(min(200, len(train_df)), random_state=0)
            )
            if not _lgbm_x.empty:
                log_shap_artifacts(lgbm_models["q50"], _lgbm_x, prefix="lgbm")
        except Exception as e:
            print(f"  [WARN] MLflow SHAP upload skipped: {e}")

        active = mlflow.active_run()
        if active:
            print(
                f"\n[MLFLOW] Run {active.info.run_id[:8]}... "
                f"logged to '{EXPERIMENTS['lgbm']}'"
            )

        # -- Write forecasts to TimeDB -----------------------------------------
        print("\nWriting forecasts to TimeDB ...")
        lgbm_full_cal = lgbm.predict(build_features(zone_cfg, cal_start, end, td=td))
        lear_full = lear.predict(build_features(zone_cfg, cal_start, end, td=td))
        _write_forecasts_to_timedb(td, lgbm_full_cal, lear_full, knowledge_time=run_kt)

        trained_info = {
            "trained_at": run_kt.isoformat(),
            "train_end": str(train_end.date()),
            "cal_start": str(cal_start.date()),
            "test_start": str(test_start.date()),
            "val_end": str(end.date()),
        }
        trained_path = MODEL_DIR / "trained_at.json"
        with open(trained_path, "w") as f:
            json.dump(trained_info, f, indent=2)
        print(f"  [OK] Training cache written -> {trained_path}")

        print("\nUploading model artifacts to S3 ...")
        _s3_upload_models()

        print(f"\n[OK] All models trained. Metrics saved to {metrics_path}")

    # -- XGBoost: separate MLflow run in nordspot-xgboost experiment -----------
    # train_df / cal_df / test_df / test_actuals / test_start are still in scope
    # from the LGBM block above - Python with-blocks do not create a new scope.
    mlflow.set_experiment(EXPERIMENTS["xgboost"])
    with mlflow.start_run(run_name=f"xgboost-{end.date()}"):
        mlflow.set_tags(
            {
                "note": note,
                "zone": zone,
                "train_start": str(start.date()),
                "run_end": str(end.date()),
            }
        )

        print("\nTraining XGBoost ...")
        xgb_models = xgb_model.train(train_df)

        # Conformal calibration on the cal window
        print("\nCalibrating XGBoost prediction intervals (split conformal) ...")
        xgb_cal_raw = xgb_model.predict(cal_df, apply_conformal=False)
        xgb_cal_actuals = cal_df["price"]
        xgb_cal_mask = xgb_cal_actuals.notna()
        xgb_model.calibrate(
            xgb_cal_actuals[xgb_cal_mask],
            xgb_cal_raw["xgb_q05"][xgb_cal_mask],
            xgb_cal_raw["xgb_q95"][xgb_cal_mask],
        )

        # Metrics on test window
        xgb_test_cal = xgb_model.predict(test_df)
        xgb_test_raw = xgb_model.predict(test_df, apply_conformal=False)
        xgb_m_raw = _compute_metrics(
            test_actuals,
            xgb_test_raw["xgb_q05"],
            xgb_test_raw["xgb_q50"],
            xgb_test_raw["xgb_q95"],
            test_from=str(test_start.date()),
            test_to=str(end.date()),
        )
        xgb_m = _compute_metrics(
            test_actuals,
            xgb_test_cal["xgb_q05"],
            xgb_test_cal["xgb_q50"],
            xgb_test_cal["xgb_q95"],
            test_from=str(test_start.date()),
            test_to=str(end.date()),
        )
        xgb_m["coverage_q5_q95_raw"] = xgb_m_raw["coverage_q5_q95"]

        # MLflow: log hyperparameters
        _xgb_params = {
            k: v
            for k, v in xgb_model._XGB_PARAMS_BASE.items()
            if k not in ("n_jobs", "verbosity")
        }
        _xgb_params["xgb_val_frac"] = xgb_model.VAL_FRAC
        _xgb_params["xgb_early_stop_rounds"] = xgb_model.EARLY_STOP_N
        mlflow.log_params(_xgb_params)

        # MLflow: log metrics
        mlflow.log_metrics(
            {
                "xgb_mae": xgb_m["mae"],
                "xgb_rmse": xgb_m["rmse"],
                "xgb_mape": xgb_m["mape"],
                "xgb_coverage": xgb_m["coverage_q5_q95"],
                "xgb_spike_mae": xgb_m["spike_mae"],
                "xgb_night_mae": xgb_m["night_mae"],
                "xgb_peak_mae": xgb_m["peak_mae"],
            }
        )

        # MLflow: log model artifacts (best-effort)
        try:
            for name, model in xgb_models.items():
                mlflow_xgb.log_model(model, f"xgb_{name}")
        except Exception as e:
            print(f"  [WARN] MLflow XGBoost artifact upload skipped: {e}")

        # MLflow: log SHAP interpretability plots (best-effort)
        try:
            from ml.models.xgboost import (
                _prep as _xgb_prep,  # local import avoids circular dep
            )

            _xgb_x, _ = _xgb_prep(
                train_df.sample(min(200, len(train_df)), random_state=0)
            )
            if not _xgb_x.empty:
                log_shap_artifacts(xgb_models["q50"], _xgb_x, prefix="xgb")
        except Exception as e:
            print(f"  [WARN] MLflow XGBoost SHAP upload skipped: {e}")

        xgb_active = mlflow.active_run()
        if xgb_active:
            print(
                f"\n[MLFLOW] XGBoost run {xgb_active.info.run_id[:8]}... "
                f"logged to '{EXPERIMENTS['xgboost']}'"
            )

    # -- CatBoost: separate MLflow run in nordspot-catboost experiment ---------
    mlflow.set_experiment(EXPERIMENTS["catboost"])
    with mlflow.start_run(run_name=f"catboost-{end.date()}"):
        mlflow.set_tags(
            {
                "note": note,
                "zone": zone,
                "train_start": str(start.date()),
                "run_end": str(end.date()),
            }
        )

        print("\nTraining CatBoost ...")
        cat_models = cat_model.train(train_df)

        # Conformal calibration on the cal window
        print("\nCalibrating CatBoost prediction intervals (split conformal) ...")
        cat_cal_raw = cat_model.predict(cal_df, apply_conformal=False)
        cat_cal_actuals = cal_df["price"]
        cat_cal_mask = cat_cal_actuals.notna()
        cat_model.calibrate(
            cat_cal_actuals[cat_cal_mask],
            cat_cal_raw["cat_q05"][cat_cal_mask],
            cat_cal_raw["cat_q95"][cat_cal_mask],
        )

        # Metrics on test window
        cat_test_cal = cat_model.predict(test_df)
        cat_test_raw = cat_model.predict(test_df, apply_conformal=False)
        cat_m_raw = _compute_metrics(
            test_actuals,
            cat_test_raw["cat_q05"],
            cat_test_raw["cat_q50"],
            cat_test_raw["cat_q95"],
            test_from=str(test_start.date()),
            test_to=str(end.date()),
        )
        cat_m = _compute_metrics(
            test_actuals,
            cat_test_cal["cat_q05"],
            cat_test_cal["cat_q50"],
            cat_test_cal["cat_q95"],
            test_from=str(test_start.date()),
            test_to=str(end.date()),
        )
        cat_m["coverage_q5_q95_raw"] = cat_m_raw["coverage_q5_q95"]

        # MLflow: log hyperparameters
        _cat_params = {
            k: v
            for k, v in cat_model._CB_PARAMS_BASE.items()
            if k not in ("thread_count", "random_seed", "verbose")
        }
        _cat_params["cat_val_frac"] = cat_model.VAL_FRAC
        _cat_params["cat_early_stop_rounds"] = cat_model.EARLY_STOP_N
        _cat_params["cat_features"] = ",".join(cat_model.CAT_FEATURE_COLS)
        mlflow.log_params(_cat_params)

        # MLflow: log metrics
        mlflow.log_metrics(
            {
                "cat_mae": cat_m["mae"],
                "cat_rmse": cat_m["rmse"],
                "cat_mape": cat_m["mape"],
                "cat_coverage": cat_m["coverage_q5_q95"],
                "cat_spike_mae": cat_m["spike_mae"],
                "cat_night_mae": cat_m["night_mae"],
                "cat_peak_mae": cat_m["peak_mae"],
            }
        )

        # MLflow: log model artifacts (best-effort)
        try:
            for name, model in cat_models.items():
                mlflow_cat.log_model(model, f"cat_{name}")
        except Exception as e:
            print(f"  [WARN] MLflow CatBoost artifact upload skipped: {e}")

        # MLflow: log SHAP interpretability plots (best-effort)
        try:
            from ml.models.catboost import (
                _prep as _cat_prep,  # local import avoids circular dep
            )

            _cat_x, _ = _cat_prep(
                train_df.sample(min(200, len(train_df)), random_state=0)
            )
            if not _cat_x.empty:
                log_shap_artifacts(cat_models["q50"], _cat_x, prefix="cat")
        except Exception as e:
            print(f"  [WARN] MLflow CatBoost SHAP upload skipped: {e}")

        cat_active = mlflow.active_run()
        if cat_active:
            print(
                f"\n[MLFLOW] CatBoost run {cat_active.info.run_id[:8]}... "
                f"logged to '{EXPERIMENTS['catboost']}'"
            )

    # -- Ensemble: separate MLflow run in nordspot-ensemble experiment ----------
    # Stacks calibrated LGBM, XGBoost, and CatBoost predictions via Ridge.
    # No new training data required - base models are already saved to disk.
    # train_df / cal_df / test_df / test_actuals / test_start are in scope
    # from the LGBM block above (Python with-blocks do not create a new scope).
    mlflow.set_experiment(EXPERIMENTS["ensemble"])
    with mlflow.start_run(run_name=f"ensemble-{end.date()}"):
        mlflow.set_tags(
            {
                "note": note,
                "zone": zone,
                "train_start": str(start.date()),
                "run_end": str(end.date()),
            }
        )

        # Generate calibrated base model predictions on cal and test windows.
        # Each predict() loads saved model files - written by the blocks above.
        print("\nGenerating base model predictions for ensemble meta-learner ...")
        base_cal_preds = pd.concat(
            [
                lgbm.predict(cal_df),  # lgbm_q05, lgbm_q50, lgbm_q95
                xgb_model.predict(cal_df),  # xgb_q05,  xgb_q50,  xgb_q95
                cat_model.predict(cal_df),  # cat_q05,  cat_q50,  cat_q95
            ],
            axis=1,
        )
        base_test_preds = pd.concat(
            [
                lgbm.predict(test_df),
                xgb_model.predict(test_df),
                cat_model.predict(test_df),
            ],
            axis=1,
        )

        print("\nTraining Ridge meta-learner (one per quantile) ...")
        ens_models = ens_model.train(base_cal_preds, cal_df["price"])

        # Evaluate on held-out test window
        ens_preds = ens_model.predict(base_test_preds)
        ens_m = _compute_metrics(
            test_actuals,
            ens_preds["ens_q05"],
            ens_preds["ens_q50"],
            ens_preds["ens_q95"],
            test_from=str(test_start.date()),
            test_to=str(end.date()),
        )

        # MLflow: log Ridge coefficients per quantile for interpretability
        for q_name, model in ens_models.items():
            try:
                for feat, coef in zip(ens_model._Q_FEATURES[q_name], model.coef_):
                    mlflow.log_param(f"ens_coef_{q_name}_{feat}", round(float(coef), 4))
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] Coefficient logging skipped for {q_name}: {exc}")

        mlflow.log_params(
            {
                "ens_meta_alpha": ens_model._RIDGE_ALPHA,
                "ens_base_models": "lgbm,xgboost,catboost",
                "ens_meta_features_per_quantile": 3,
            }
        )

        mlflow.log_metrics(
            {
                "ens_mae": ens_m["mae"],
                "ens_rmse": ens_m["rmse"],
                "ens_mape": ens_m["mape"],
                "ens_coverage": ens_m["coverage_q5_q95"],
                "ens_spike_mae": ens_m["spike_mae"],
                "ens_night_mae": ens_m["night_mae"],
                "ens_peak_mae": ens_m["peak_mae"],
            }
        )

        # MLflow: log ensemble q50 meta-model artifact for Model Registry (best-effort)
        try:
            mlflow_sklearn.log_model(ens_models["q50"], "ensemble_q50")
        except Exception as e:
            print(f"  [WARN] MLflow ensemble artifact upload skipped: {e}")

        # Capture run_id before the context manager closes
        ens_run_id = mlflow.active_run().info.run_id

        ens_active = mlflow.active_run()
        if ens_active:
            print(
                f"\n[MLFLOW] Ensemble run {ens_active.info.run_id[:8]}... "
                f"logged to '{EXPERIMENTS['ensemble']}'"
            )

    # -- Model Registry: auto-promote if MAE improves --------------------------
    # Runs outside the MLflow with-block - no active run required.
    # Model name is zone-specific so each zone has its own registry lineage.
    print("\nChecking Model Registry for auto-promotion ...")
    register_and_promote(
        ens_run_id,
        ens_m["mae"],
        model_name=f"nordspot-ensemble-{zone}",
    )


# -- Multi-zone convenience wrapper --------------------------------------------


def train_all_zones(
    start: datetime,
    end: datetime,
    note: str = "Routine training run",
) -> None:
    """Train all four Swedish bidding zones in sequence (SE1, SE2, SE3, SE4).

    Each zone runs as an independent call to train(), producing its own set of
    MLflow runs (lgbm / xgboost / catboost / ensemble) and its own Model Registry
    entry (nordspot-ensemble-SE1, ..., nordspot-ensemble-SE4).
    """
    for zone_id in load_all_zones().keys():
        print(f"\n{'=' * 64}")
        print(f"  Training zone {zone_id} ...")
        print(f"{'=' * 64}")
        train(start=start, end=end, zone=zone_id, note=note)


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
        help="Short description of changes made (logged to MODEL_LOG.md and MLflow)",
    )
    parser.add_argument(
        "--zone",
        default="SE3",
        choices=["SE1", "SE2", "SE3", "SE4", "ALL"],
        help="Bidding zone to train (default SE3). Pass ALL to train all four zones.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if the model cache is fresh (< 7 days old).",
    )
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    if not args.force:
        _td = init_schema()
        if _should_skip_training(_td):
            import sys

            sys.exit(0)

    if args.zone == "ALL":
        train_all_zones(start, end, note=args.note)
    else:
        train(start, end, zone=args.zone, note=args.note)
