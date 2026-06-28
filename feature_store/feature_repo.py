"""NordSpot Feast feature repository.

Defines all entities, feature views, and the feature service.
This file is the single source of truth for the Gold feature layer.

Data flow:
    pipeline/gold_writer.py  →  data/gold/features/{zone}/{year}/{month}/*.parquet
                             →  FileSource (this file)
                             →  Feast offline store  →  model training
                             →  Feast online store   →  FastAPI serving

Source convention (Gold Parquet):
    Path: <project_root>/data/gold/features/{zone_eic}/{year}/{month:02d}/*.parquet
    Columns: valid_time (UTC datetime), zone (EIC str), + all feature columns

⚠️  Run ``feast apply`` only after Gold Parquet files exist (Story 3.3).
    Until then, definitions here can be imported and tested as plain Python objects.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

# ── Paths ──────────────────────────────────────────────────────────────────────
# __file__ is feature_store/feature_repo.py → parent is feature_store/ → parent.parent is project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_ROOT = _PROJECT_ROOT / "data" / "gold" / "features"

# ── Entity ─────────────────────────────────────────────────────────────────────
zone = Entity(
    name="zone",
    join_keys=["zone"],
    value_type=ValueType.STRING,
    description="ENTSO-E EIC bidding zone code (e.g. 10Y1001A1001A46L for SE3)",
)

# ── Shared data source ─────────────────────────────────────────────────────────
# All three feature views read from the same Gold-layer Parquet tree.
# Feast selects only the columns declared in each view's schema.
_gold_source = FileSource(
    name="nordspot_gold_features",
    path=str(GOLD_ROOT / "**" / "*.parquet"),
    timestamp_field="valid_time",
)

# ── Feature view 1: Price lags & rolling stats ─────────────────────────────────
price_features = FeatureView(
    name="price_features",
    entities=[zone],
    ttl=timedelta(days=30),
    schema=[
        # Autoregressive lags — all safe at D-1 noon gate closure
        Field(name="price_lag23h", dtype=Float64),
        Field(name="price_lag24h", dtype=Float64),
        Field(name="price_lag25h", dtype=Float64),
        Field(name="price_lag48h", dtype=Float64),
        Field(name="price_lag72h", dtype=Float64),
        Field(name="price_lag168h", dtype=Float64),
        Field(name="price_lag336h", dtype=Float64),
        # Rolling statistics (all shifted 24h to prevent leakage)
        Field(name="price_roll24h_mean", dtype=Float64),
        Field(name="price_roll168h_mean", dtype=Float64),
        Field(name="price_roll168h_std", dtype=Float64),
        Field(name="price_roll720h_std", dtype=Float64),
    ],
    source=_gold_source,
    online=True,
)

# ── Feature view 2: Calendar ───────────────────────────────────────────────────
calendar_features = FeatureView(
    name="calendar_features",
    entities=[zone],
    ttl=timedelta(days=30),
    schema=[
        # Raw calendar fields
        Field(name="hour", dtype=Int64),
        Field(name="weekday", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="is_holiday", dtype=Int64),
        Field(name="hour_of_week", dtype=Int64),
        # Cyclical encoding — preserves circular distance (hour 23 ≈ hour 0)
        Field(name="hour_sin", dtype=Float64),
        Field(name="hour_cos", dtype=Float64),
        Field(name="weekday_sin", dtype=Float64),
        Field(name="weekday_cos", dtype=Float64),
        Field(name="month_sin", dtype=Float64),
        Field(name="month_cos", dtype=Float64),
        # Interactions
        Field(name="hour_x_month", dtype=Float64),
        Field(name="weekend_x_hour", dtype=Float64),
    ],
    source=_gold_source,
    online=True,
)

# ── Feature view 3: Weather, generation, load, cross-border ───────────────────
grid_weather_features = FeatureView(
    name="grid_weather_features",
    entities=[zone],
    ttl=timedelta(days=30),
    schema=[
        # Weather — from Open-Meteo historical forecast archive (see PLATFORM_PLAN.md)
        Field(name="temperature_2m", dtype=Float64),
        Field(name="wind_speed_10m", dtype=Float64),
        Field(name="solar_radiation", dtype=Float64),
        Field(name="temp_x_wind", dtype=Float64),
        Field(name="temp_x_hour", dtype=Float64),
        # Generation (from Silver ClickHouse)
        Field(name="wind_mw", dtype=Float64),
        Field(name="solar_mw", dtype=Float64),
        # Load
        Field(name="load_mw", dtype=Float64),
        # Cross-border net exchange (imports − exports)
        Field(name="net_exchange_mw", dtype=Float64),
    ],
    source=_gold_source,
    online=True,
)

# ── Feature service ────────────────────────────────────────────────────────────
# Single service used by both training (offline) and serving (online).
# Training: store.get_historical_features(entity_df, features=nordspot_forecast_service)
# Serving:  store.get_online_features(entity_rows, features=nordspot_forecast_service)
nordspot_forecast_service = FeatureService(
    name="nordspot_forecast_features",
    features=[price_features, calendar_features, grid_weather_features],
    description="Full feature set for NordSpot day-ahead price forecasting (SE1–SE4)",
)
