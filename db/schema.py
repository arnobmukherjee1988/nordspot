"""Series-ID registry and schema initialisation for SE3 forecast project.

TimeDB stores every time series by integer series_id.  We manage the
mapping here so the rest of the codebase never hard-codes raw integers.
"""

from timedb import TimeDBClient

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


if __name__ == "__main__":
    td = init_schema()
    print("Schema ready.  Series registry:")
    for name, sid in SERIES.items():
        print(f"  {sid:>3}  {name}")
