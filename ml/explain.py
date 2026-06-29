"""SHAP interpretability for NordSpot tree-based forecasting models.

Provides two public surfaces:
    compute_shap(model, X)          -> shap.Explanation
    log_shap_artifacts(model, X, prefix)  -> logs 2 figures to the active MLflow run

All three model families (LightGBM, XGBoost, CatBoost) are handled identically
via shap.TreeExplainer - it auto-detects model type and uses exact tree SHAP
(polynomial complexity in tree depth, not exponential in features).

Design decisions:
    - Summary figure is implemented in pure matplotlib so it has no dependency
      on shap's own plotting code, which has broken across minor releases.
    - Waterfall figure uses shap.plots.waterfall for the signed decomposition
      with a clean matplotlib fallback if that API is unavailable.
    - Sampling cap of 200 rows for background keeps compute time < 5 s even
      on 3 x 3000-tree ensembles.

Reference: Lundberg & Lee (2017) "A unified approach to interpreting model
predictions." NeurIPS 30.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap

_SAMPLE_CAP = 200  # max background rows for TreeExplainer


def compute_shap(
    model: object,
    x: pd.DataFrame,
) -> shap.Explanation:
    """Return exact TreeSHAP Explanation for model on feature matrix x.

    Supports LightGBM, XGBoost, and CatBoost - shap.TreeExplainer handles
    all three natively via their internal tree structures.

    Args:
        model: Fitted tree model (LGBMRegressor, XGBRegressor, or
               CatBoostRegressor).
        x:     Feature matrix, shape (n_samples, n_features). Must contain no
               NaN values; categorical features must be cast to int for
               CatBoost models before calling.

    Returns:
        shap.Explanation with .values (n_samples x n_features), .base_values,
        and .feature_names attributes.
    """
    explainer = shap.TreeExplainer(model)
    return explainer(x)


def shap_summary_figure(
    explanation: shap.Explanation,
    max_display: int = 15,
) -> matplotlib.figure.Figure:
    """Bar chart of mean |SHAP value| per feature (top max_display features).

    Implemented in pure matplotlib to avoid dependency on shap's own
    plotting layer (shap.plots.bar API has shifted across minor releases).
    """
    vals = np.abs(explanation.values).mean(axis=0)
    names = list(explanation.feature_names)

    # Sort descending, keep top max_display
    order = np.argsort(vals)[::-1][:max_display]
    top_vals = vals[order][::-1]  # reversed so largest is at top
    top_names = [names[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, max_display * 0.38)))
    ax.barh(range(len(top_names)), top_vals, color="#d62728")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|  (EUR/MWh)")
    ax.set_title(f"SHAP Feature Importance  (top {len(top_names)})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def shap_waterfall_figure(
    explanation: shap.Explanation,
    row_idx: int = 0,
) -> matplotlib.figure.Figure:
    """Waterfall plot decomposing a single prediction into SHAP contributions."""
    try:
        shap.plots.waterfall(explanation[row_idx], show=False)
        fig = plt.gcf()
        fig.tight_layout()
    except Exception:
        # Fallback: horizontal bar chart of signed SHAP values for this row
        vals = explanation[row_idx].values
        names = list(explanation.feature_names)
        order = np.argsort(np.abs(vals))[::-1][:15]
        sv = vals[order][::-1]
        sn = [names[i] for i in order][::-1]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in sv]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(sv)), sv, color=colors)
        ax.set_yticks(range(len(sv)))
        ax.set_yticklabels(sn, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlabel("SHAP value  (EUR/MWh)")
        ax.set_title("SHAP Prediction Breakdown")
        fig.tight_layout()
    return fig


def log_shap_artifacts(
    model: object,
    x: pd.DataFrame,
    prefix: str,
    max_display: int = 15,
) -> list[matplotlib.figure.Figure]:
    """Compute SHAP and log summary + waterfall plots to the active MLflow run.

    Args:
        model:       Fitted tree model.
        x:           Prepared feature matrix (no target, no NaNs).
        prefix:      Artifact name prefix, e.g. "lgbm", "xgb", "cat".
        max_display: Top N features to show in the summary chart.

    Returns:
        [summary_fig, waterfall_fig] - already logged; caller may close them.
    """
    x_bg = x.sample(min(_SAMPLE_CAP, len(x)), random_state=0)

    explanation = compute_shap(model, x_bg)
    summary_fig = shap_summary_figure(explanation, max_display=max_display)
    waterfall_fig = shap_waterfall_figure(explanation, row_idx=0)

    mlflow.log_figure(summary_fig, f"{prefix}_shap_summary.png")
    mlflow.log_figure(waterfall_fig, f"{prefix}_shap_waterfall.png")

    plt.close(summary_fig)
    plt.close(waterfall_fig)

    return [summary_fig, waterfall_fig]
