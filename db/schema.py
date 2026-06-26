"""Series-ID registry and schema initialisation for NordSpot.

Two storage layers:
  - TimeDB (integer series_id): prices, weather, ML forecast quantiles.
  - ClickHouse native tables (zone-aware): generation, load, cross-border flows.
    These use clickhouse-connect and are created via create_zone_tables().
"""

from timedb import TimeDBClient

# ── ClickHouse DDL — zone-aware Silver tables ─────────────────────────────────
# All tables: MergeTree, partitioned by month, ordered by (zone, valid_time).
# CREATE TABLE IF NOT EXISTS makes every statement idempotent.

ZONE_TABLE_DDL: dict[str, str] = {
    "generation_actual": """
        CREATE TABLE IF NOT EXISTS generation_actual (
            valid_time DateTime,
            zone       LowCardinality(String),
            total_mw   Float64,
            wind_mw    Float64,
            solar_mw   Float64
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(valid_time)
        ORDER BY (zone, valid_time)
    """,
    "load_actual": """
        CREATE TABLE IF NOT EXISTS load_actual (
            valid_time DateTime,
            zone       LowCardinality(String),
            value_mw   Float64
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(valid_time)
        ORDER BY (zone, valid_time)
    """,
    "crossborder_flows": """
        CREATE TABLE IF NOT EXISTS crossborder_flows (
            valid_time DateTime,
            from_zone  LowCardinality(String),
            to_zone    LowCardinality(String),
            value_mw   Float64
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(valid_time)
        ORDER BY (from_zone, to_zone, valid_time)
    """,
}

# ── Series ID registry ────────────────────────────────────────────────────────
# Add new series here as the project grows; never reuse an ID.

SERIES = {
    # Raw ENTSO-E day-ahead prices (EUR/MWh, hourly)
    "prices_raw": 1,
    # Weather features (Open-Meteo, hourly)
    "weather_temperature": 10,
    "weather_wind_speed": 11,
    "weather_irradiance": 12,
    # ENTSO-E generation (actual, hourly, MW)
    "generation_total": 40,
    "generation_wind": 41,
    "generation_solar": 42,
    # ENTSO-E actual load (MW, hourly)
    "load_actual": 50,
    # Cross-border physical flows (MW, hourly) — one ID per major zone pair
    "crossborder_SE1_SE2": 60,
    "crossborder_SE2_SE3": 61,
    "crossborder_SE3_SE4": 62,
    # LightGBM forecast quantiles
    "lgbm_q05": 20,
    "lgbm_q50": 21,
    "lgbm_q95": 22,
    # LEAR forecast quantiles
    "lear_q05": 30,
    "lear_q50": 31,
    "lear_q95": 32,
}


def init_schema(ch_url: str | None = None) -> TimeDBClient:
    """Create TimeDB tables (idempotent — safe to call on every startup)."""
    td = TimeDBClient(ch_url=ch_url)
    td.create()
    return td


def create_zone_tables() -> list[str]:
    """Create zone-aware ClickHouse tables (idempotent).

    Executes each DDL in ZONE_TABLE_DDL via clickhouse-connect.
    Returns the list of table names that were processed.
    """
    from db.clickhouse import get_client

    client = get_client()
    created = []
    for table_name, ddl in ZONE_TABLE_DDL.items():
        client.command(ddl.strip())
        created.append(table_name)
    return created


if __name__ == "__main__":
    td = init_schema()
    print("TimeDB schema ready.  Series registry:")
    for name, sid in SERIES.items():
        print(f"  {sid:>3}  {name}")
    print()
    tables = create_zone_tables()
    print(f"Zone-aware tables created: {', '.join(tables)}")
