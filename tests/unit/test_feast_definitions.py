"""Unit tests for Feast entity and feature view definitions.

Tests validate the Python object definitions — names, schema fields, TTL,
entity join keys, shared source — WITHOUT requiring ``feast apply`` or any
running infrastructure (no registry, no ClickHouse, no Parquet files).

Strategy: load feast/feature_repo.py via importlib so we avoid putting
feast/ on sys.path (which would shadow the installed feast package).
"""

from __future__ import annotations

import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest

from feast import Entity, FeatureService, FeatureView
from feast.types import Float64, Int64

FEAST_DIR = Path(__file__).parents[2] / "feature_store"


def _load_module(name: str):
    """Load a module from feast/ by filename without modifying sys.path."""
    path = FEAST_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def repo():
    return _load_module("feature_repo")


# ── Entity ──────────────────────────────────────────────────────────────────────


def test_zone_entity_type(repo):
    assert isinstance(repo.zone, Entity)


def test_zone_entity_join_key(repo):
    # Feast 0.64 exposes the primary join key as .join_key (singular)
    assert repo.zone.join_key == "zone"


def test_zone_entity_name(repo):
    assert repo.zone.name == "zone"


# ── Shared source ───────────────────────────────────────────────────────────────


def test_source_timestamp_field(repo):
    assert repo._gold_source.timestamp_field == "valid_time"


def test_source_name(repo):
    assert repo._gold_source.name == "nordspot_gold_features"


def test_all_views_share_same_source(repo):
    """All three feature views must reference the same FileSource."""
    sources = {
        repo.price_features.source.name,
        repo.calendar_features.source.name,
        repo.grid_weather_features.source.name,
    }
    assert len(sources) == 1, f"Expected 1 unique source, got: {sources}"


# ── Price features ──────────────────────────────────────────────────────────────


def test_price_features_is_feature_view(repo):
    assert isinstance(repo.price_features, FeatureView)


def test_price_features_name(repo):
    assert repo.price_features.name == "price_features"


def test_price_features_ttl(repo):
    assert repo.price_features.ttl == timedelta(days=30)


def test_price_features_has_all_lag_fields(repo):
    names = {f.name for f in repo.price_features.schema}
    for lag in [23, 24, 25, 48, 72, 168, 336]:
        assert f"price_lag{lag}h" in names, f"Missing price_lag{lag}h"


def test_price_features_has_rolling_stats(repo):
    names = {f.name for f in repo.price_features.schema}
    for col in (
        "price_roll24h_mean",
        "price_roll168h_mean",
        "price_roll168h_std",
        "price_roll720h_std",
    ):
        assert col in names, f"Missing {col}"


def test_price_features_all_float64(repo):
    for field in repo.price_features.schema:
        assert (
            field.dtype == Float64
        ), f"{field.name} should be Float64, got {field.dtype}"


# ── Calendar features ───────────────────────────────────────────────────────────


def test_calendar_features_is_feature_view(repo):
    assert isinstance(repo.calendar_features, FeatureView)


def test_calendar_features_name(repo):
    assert repo.calendar_features.name == "calendar_features"


def test_calendar_features_has_cyclical_cols(repo):
    names = {f.name for f in repo.calendar_features.schema}
    for col in (
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
    ):
        assert col in names, f"Missing cyclical column: {col}"


def test_calendar_integer_fields_are_int64(repo):
    int_cols = {"hour", "weekday", "month", "is_weekend", "is_holiday", "hour_of_week"}
    for field in repo.calendar_features.schema:
        if field.name in int_cols:
            assert (
                field.dtype == Int64
            ), f"{field.name} should be Int64, got {field.dtype}"


def test_calendar_has_interaction_cols(repo):
    names = {f.name for f in repo.calendar_features.schema}
    assert "hour_x_month" in names
    assert "weekend_x_hour" in names


# ── Grid / weather features ─────────────────────────────────────────────────────


def test_grid_weather_is_feature_view(repo):
    assert isinstance(repo.grid_weather_features, FeatureView)


def test_grid_weather_name(repo):
    assert repo.grid_weather_features.name == "grid_weather_features"


def test_grid_weather_has_weather_cols(repo):
    names = {f.name for f in repo.grid_weather_features.schema}
    for col in (
        "temperature_2m",
        "wind_speed_10m",
        "solar_radiation",
        "temp_x_wind",
        "temp_x_hour",
    ):
        assert col in names, f"Missing weather column: {col}"


def test_grid_weather_has_grid_cols(repo):
    names = {f.name for f in repo.grid_weather_features.schema}
    for col in ("wind_mw", "solar_mw", "load_mw", "net_exchange_mw"):
        assert col in names, f"Missing grid column: {col}"


def test_grid_weather_all_float64(repo):
    for field in repo.grid_weather_features.schema:
        assert field.dtype == Float64, f"{field.name} should be Float64"


# ── Feature service ─────────────────────────────────────────────────────────────


def test_forecast_service_is_feature_service(repo):
    assert isinstance(repo.nordspot_forecast_service, FeatureService)


def test_forecast_service_name(repo):
    assert repo.nordspot_forecast_service.name == "nordspot_forecast_features"


def test_forecast_service_includes_all_three_views(repo):
    view_names = {
        p.name for p in repo.nordspot_forecast_service.feature_view_projections
    }
    assert "price_features" in view_names
    assert "calendar_features" in view_names
    assert "grid_weather_features" in view_names


def test_all_views_have_consistent_ttl(repo):
    """All views should share the same TTL to simplify materialisation."""
    ttls = {
        repo.price_features.ttl,
        repo.calendar_features.ttl,
        repo.grid_weather_features.ttl,
    }
    assert len(ttls) == 1, f"TTL mismatch across views: {ttls}"
