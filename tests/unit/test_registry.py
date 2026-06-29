"""Unit tests for ml/registry.py.

register_and_promote() is tested against a mocked MlflowClient so no real
tracking server, artifact store, or model file is required.

Four scenarios are covered:
  1. No Production version exists          -> unconditional promotion
  2. New MAE < Production MAE              -> promotion, old version archived
  3. New MAE >= Production MAE              -> Staging (challenger)
  4. Version is tagged with the correct MAE value
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

from ml.registry import register_and_promote

# -- Helper --------------------------------------------------------------------


def _make_client(prod_mae: float | None = None) -> MagicMock:
    """Return a mock MlflowClient with an optional Production version.

    Args:
        prod_mae: MAE tag on the existing Production version. Pass None to
                  simulate the registry having no Production version yet.
    """
    client = MagicMock()

    # create_model_version -> version "2" in READY state
    new_mv = MagicMock()
    new_mv.version = "2"
    new_mv.status = "READY"
    client.create_model_version.return_value = new_mv
    client.get_model_version.return_value = new_mv  # for READY polling

    # get_latest_versions("Production") -> empty or one prod version
    if prod_mae is None:
        client.get_latest_versions.return_value = []
    else:
        prod_mv = MagicMock()
        prod_mv.version = "1"
        prod_mv.tags = {"mae": str(prod_mae)}
        client.get_latest_versions.return_value = [prod_mv]

    return client


# -- Tests ---------------------------------------------------------------------


def test_promotes_when_no_production_exists():
    """First run: no Production version -> unconditional promotion."""
    client = _make_client(prod_mae=None)
    with patch("ml.registry.MlflowClient", return_value=client):
        result = register_and_promote(run_id="abc123", mae=12.0)

    assert result["action"] == "promoted"
    assert result["mae"] == 12.0
    client.transition_model_version_stage.assert_called_with(
        "nordspot-ensemble", "2", "Production"
    )


def test_promotes_when_new_mae_is_better():
    """New MAE (12.0) < Production MAE (15.0) -> promote new, archive old."""
    client = _make_client(prod_mae=15.0)
    with patch("ml.registry.MlflowClient", return_value=client):
        result = register_and_promote(run_id="abc123", mae=12.0)

    assert result["action"] == "promoted"
    assert result["prev_mae"] == 15.0

    calls = client.transition_model_version_stage.call_args_list
    assert (
        call("nordspot-ensemble", "1", "Archived") in calls
    ), "Old Production version should be Archived"
    assert (
        call("nordspot-ensemble", "2", "Production") in calls
    ), "New version should be promoted to Production"


def test_stays_challenger_when_new_mae_is_worse():
    """New MAE (18.0) >= Production MAE (15.0) -> Staging (challenger)."""
    client = _make_client(prod_mae=15.0)
    with patch("ml.registry.MlflowClient", return_value=client):
        result = register_and_promote(run_id="abc123", mae=18.0)

    assert result["action"] == "challenger"
    assert result["prod_mae"] == 15.0
    client.transition_model_version_stage.assert_called_with(
        "nordspot-ensemble", "2", "Staging"
    )


def test_new_version_is_tagged_with_mae():
    """set_model_version_tag must be called with key='mae' and the rounded value."""
    client = _make_client(prod_mae=None)
    with patch("ml.registry.MlflowClient", return_value=client):
        register_and_promote(run_id="deadbeef", mae=9.1234)

    tag_calls = client.set_model_version_tag.call_args_list
    mae_call = next(
        (c for c in tag_calls if len(c.args) >= 3 and c.args[2] == "mae"),
        None,
    )
    assert mae_call is not None, "set_model_version_tag should be called with key 'mae'"
    assert (
        mae_call.args[3] == "9.1234"
    ), f"MAE tag should be '9.1234', got '{mae_call.args[3]}'"
