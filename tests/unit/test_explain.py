"""Unit tests for ml/explain.py.

Verifies SHAP value properties (shape, additivity) and that
log_shap_artifacts calls mlflow.log_figure with the expected artifact names.

Uses a tiny LightGBM model (10 trees, 10 features) - fast to train, and
sufficient to exercise the full TreeExplainer code path.
"""

from __future__ import annotations

from unittest.mock import patch

import lightgbm as lgb
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ml.explain import (
    compute_shap,
    log_shap_artifacts,
    shap_summary_figure,
)
from ml.models.lgbm import FEATURE_COLS

# Use only a subset of features to keep the fixture lightweight
_FEAT_SUBSET = FEATURE_COLS[:10]
_N = 100


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture()
def tiny_model_and_x() -> tuple[lgb.LGBMRegressor, pd.DataFrame]:
    """Train a 10-tree LGBM model on 100 synthetic rows."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {col: rng.standard_normal(_N) for col in _FEAT_SUBSET},
    )
    y = rng.uniform(20, 200, _N)
    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    model.fit(x, y)
    return model, x


# -- Tests ---------------------------------------------------------------------


def test_compute_shap_shape(tiny_model_and_x):
    """SHAP values must have shape (n_samples, n_features)."""
    model, x = tiny_model_and_x
    explanation = compute_shap(model, x)
    assert explanation.values.shape == (_N, len(_FEAT_SUBSET))


def test_compute_shap_additivity(tiny_model_and_x):
    """SHAP additivity: sum(shap_values[i]) + base_value == prediction[i]."""
    model, x = tiny_model_and_x
    explanation = compute_shap(model, x)
    reconstructed = explanation.values.sum(axis=1) + explanation.base_values
    actual = model.predict(x)
    np.testing.assert_allclose(reconstructed, actual, rtol=1e-4)


def test_shap_summary_figure_type(tiny_model_and_x):
    """shap_summary_figure must return a matplotlib Figure."""
    model, x = tiny_model_and_x
    explanation = compute_shap(model, x)
    fig = shap_summary_figure(explanation, max_display=5)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_log_shap_artifacts_logs_two_named_files(tiny_model_and_x):
    """log_shap_artifacts must call mlflow.log_figure exactly twice with
    filenames '<prefix>_shap_summary.png' and '<prefix>_shap_waterfall.png'.
    """
    model, x = tiny_model_and_x
    with patch("ml.explain.mlflow.log_figure") as mock_log:
        figs = log_shap_artifacts(model, x, prefix="test")

    assert mock_log.call_count == 2
    logged_names = {c.args[1] for c in mock_log.call_args_list}
    assert logged_names == {"test_shap_summary.png", "test_shap_waterfall.png"}
    assert len(figs) == 2
