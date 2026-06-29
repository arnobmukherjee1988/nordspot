"""Unit tests for ml/registry.py.

register_and_promote() is tested against a mocked MlflowClient so no real
tracking server, artifact store, or model file is required.

Four scenarios are covered:
  1. No champion version exists          -> unconditional promotion
  2. New MAE < champion MAE              -> promotion, old alias removed
  3. New MAE >= champion MAE             -> challenger alias assigned
  4. Version is tagged with the correct MAE value
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import mlflow

from ml.registry import register_and_promote

# -- Helper --------------------------------------------------------------------


def _make_client(champion_mae: float | None = None) -> MagicMock:
    """Return a mock MlflowClient with an optional champion version.

    Args:
        champion_mae: MAE tag on the existing champion version. Pass None to
                      simulate the registry having no champion version yet.
    """
    client = MagicMock()

    # create_model_version -> version "2" in READY state
    new_mv = MagicMock()
    new_mv.version = "2"
    new_mv.status = "READY"
    client.create_model_version.return_value = new_mv
    client.get_model_version.return_value = new_mv  # for READY polling

    # get_model_version_by_alias("champion") -> raises or returns champion
    if champion_mae is None:
        client.get_model_version_by_alias.side_effect = (
            mlflow.exceptions.MlflowException("No alias")
        )
    else:
        champ_mv = MagicMock()
        champ_mv.version = "1"
        champ_mv.tags = {"mae": str(champion_mae)}
        client.get_model_version_by_alias.return_value = champ_mv

    return client


# -- Tests ---------------------------------------------------------------------


def test_promotes_when_no_champion_exists():
    """First run: no champion version -> unconditional promotion."""
    client = _make_client(champion_mae=None)
    with patch("ml.registry.MlflowClient", return_value=client):
        result = register_and_promote(run_id="abc123", mae=12.0)

    assert result["action"] == "promoted"
    assert result["mae"] == 12.0
    client.set_registered_model_alias.assert_called_with(
        "nordspot-ensemble", "champion", "2"
    )


def test_promotes_when_new_mae_is_better():
    """New MAE (12.0) < champion MAE (15.0) -> promote new, remove old alias."""
    client = _make_client(champion_mae=15.0)
    with patch("ml.registry.MlflowClient", return_value=client):
        result = register_and_promote(run_id="abc123", mae=12.0)

    assert result["action"] == "promoted"
    assert result["prev_mae"] == 15.0

    alias_calls = client.set_registered_model_alias.call_args_list
    delete_calls = client.delete_registered_model_alias.call_args_list
    assert (
        call("nordspot-ensemble", "champion", "2") in alias_calls
    ), "New version should get champion alias"
    assert (
        call("nordspot-ensemble", "champion") in delete_calls
    ), "Old champion alias should be removed"


def test_stays_challenger_when_new_mae_is_worse():
    """New MAE (18.0) >= champion MAE (15.0) -> challenger alias assigned."""
    client = _make_client(champion_mae=15.0)
    with patch("ml.registry.MlflowClient", return_value=client):
        result = register_and_promote(run_id="abc123", mae=18.0)

    assert result["action"] == "challenger"
    assert result["prod_mae"] == 15.0
    client.set_registered_model_alias.assert_called_with(
        "nordspot-ensemble", "challenger", "2"
    )


def test_new_version_is_tagged_with_mae():
    """set_model_version_tag must be called with key='mae' and the rounded value."""
    client = _make_client(champion_mae=None)
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
