"""Feature engineering for NordSpot day-ahead price forecasting.

Builds an hourly feature matrix from Silver-layer data for a given zone
and date range.  All features are temporally safe at day-ahead gate closure
(NordPool closes at 12:00 CET on day D for day D+1 delivery).

Feature groups
--------------
1. Price lags & rolling stats   — autoregressive + volatility context
2. Calendar                     — hour, weekday, month, holiday, cyclical encoding
3. Weather                      — temperature, wind, solar irradiance + interactions
4. Generation                   — wind MW, solar MW from Silver (ClickHouse)
5. Load                         — actual load MW from Silver (ClickHouse)
6. Cross-border                 — net exchange position from Silver (ClickHouse)

⚠️  DESIGN CONSTRAINT (see PLATFORM_PLAN.md Epic 3):
    Weather features at training time must come from the Open-Meteo historical
    *forecast* archive, not the observation archive.  At inference time, use
    the live forecast API.  Both must be the same signal type to avoid
    training-serving skew.
"""

from __future__ import annotations

from datetime import datetime, timezone

import holidays
import numpy as np
import pandas as pd

from config.zone_config import ZoneConfig
from db.schema import SERIES

# ── TimeDB helpers ────────────────────────────────────────────────────────────


def _read_series(td, series_id: int) -> pd.Series:
    """Pull a single series from TimeDB → hourly pandas Series (UTC index)."""
    df = td.read(series_ids=[series_id], retention="forever")
    if len(df) == 0:
        return pd.Series(dtype=float, name=series_id)
    pdf = df.to_pandas()
    pdf["valid_time"] = pd.to_datetime(pdf["valid_time"], utc=True)
    return pdf.set_index("valid_time")["value"].sort_index()


# ── Pure transformation helpers (all testable without DB) ────────────────────


def _build_price_lags(price: pd.Series) -> list[pd.Series]:
    """Lag features.  All safe at day-ahead gate (noon D-1):

    - 23h, 25h  adjacent-hour prices from yesterday (cross-hour context)
    - 24h, 48h  same hour 1 and 2 days ago
    - 72h        same hour 3 days ago (Mon/Fri pattern)
    - 168h       same hour 1 week ago
    - 336h       same hour 2 weeks ago (fortnightly nuclear outage cycles)
    """
    lags = [23, 24, 25, 48, 72, 168, 336]
    return [price.shift(h).rename(f"price_lag{h}h") for h in lags]


def _build_rolling_stats(price: pd.Series) -> list[pd.Series]:
    """Rolling mean/std shifted 24h to prevent leakage."""
    base = price.shift(24).rename("price")
    return [
        base.rolling(24, min_periods=1).mean().rename("price_roll24h_mean"),
        base.rolling(168, min_periods=1).mean().rename("price_roll168h_mean"),
        base.rolling(168, min_periods=1).std().rename("price_roll168h_std"),
        base.rolling(720, min_periods=1).std().rename("price_roll720h_std"),
    ]


def _build_calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features with cyclical encoding."""
    _se_holidays = holidays.Sweden()  # lazy init — keeps module-level import clean
    cal = pd.DataFrame(index=idx)
    cal["hour"] = idx.hour
    cal["weekday"] = idx.dayofweek
    cal["month"] = idx.month
    cal["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    cal["is_holiday"] = idx.normalize().map(lambda d: int(d.date() in _se_holidays))
    cal["hour_of_week"] = idx.dayofweek * 24 + idx.hour

    # Cyclical encoding — preserves circular distance (e.g. hour 23 ≈ hour 0)
    cal["hour_sin"] = np.sin(2 * np.pi * cal["hour"] / 24)
    cal["hour_cos"] = np.cos(2 * np.pi * cal["hour"] / 24)
    cal["weekday_sin"] = np.sin(2 * np.pi * cal["weekday"] / 7)
    cal["weekday_cos"] = np.cos(2 * np.pi * cal["weekday"] / 7)
    cal["month_sin"] = np.sin(2 * np.pi * (cal["month"] - 1) / 12)
    cal["month_cos"] = np.cos(2 * np.pi * (cal["month"] - 1) / 12)

    # Interactions
    cal["hour_x_month"] = cal["hour"] * cal["month"]
    cal["weekend_x_hour"] = cal["is_weekend"] * cal["hour_sin"]
    return cal


def _build_weather_interactions(
    temp: pd.Series,
    wind: pd.Series,
    hour: pd.Series,
) -> list[pd.Series]:
    """Interaction features between weather and time-of-day."""
    return [
        (temp * wind).rename("temp_x_wind"),
        (temp * hour).rename("temp_x_hour"),
    ]


# ── ClickHouse Silver readers ─────────────────────────────────────────────────


def _ch_query_df(ch_client, sql: str) -> pd.DataFrame:
    """Execute SQL via clickhouse-connect and return a pandas DataFrame."""
    return ch_client.query_df(sql)


_EMPTY_DTI = pd.DatetimeIndex([], tz="UTC", name="valid_time")


def _read_generation(
    ch_client, zone_eic: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """Read wind_mw and solar_mw from silver_generation for [start, end)."""
    sql = f"""
        SELECT valid_time, wind_mw, solar_mw
        FROM nordspot.silver_generation
        WHERE zone = '{zone_eic}'
          AND valid_time >= '{start.strftime("%Y-%m-%d %H:%M:%S")}'
          AND valid_time <  '{end.strftime("%Y-%m-%d %H:%M:%S")}'
        ORDER BY valid_time
    """
    df = _ch_query_df(ch_client, sql)
    if df.empty:
        # Return with correct index type so .reindex(idx) works downstream
        return pd.DataFrame(
            {"wind_mw": pd.Series(dtype=float), "solar_mw": pd.Series(dtype=float)},
            index=_EMPTY_DTI,
        )
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    return df.set_index("valid_time")


def _read_load(
    ch_client, zone_eic: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """Read actual load from silver_load for [start, end)."""
    sql = f"""
        SELECT valid_time, value_mw AS load_mw
        FROM nordspot.silver_load
        WHERE zone = '{zone_eic}'
          AND valid_time >= '{start.strftime("%Y-%m-%d %H:%M:%S")}'
          AND valid_time <  '{end.strftime("%Y-%m-%d %H:%M:%S")}'
        ORDER BY valid_time
    """
    df = _ch_query_df(ch_client, sql)
    if df.empty:
        return pd.DataFrame({"load_mw": pd.Series(dtype=float)}, index=_EMPTY_DTI)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    return df.set_index("valid_time")


def _read_net_exchange(
    ch_client, zone_eic: str, start: datetime, end: datetime
) -> pd.Series:
    """Compute net exchange position for a zone.

    net_exchange = imports_into_zone - exports_from_zone
    Positive  → zone is a net importer (tighter local supply → higher price).
    Negative  → zone is a net exporter.
    """
    start_s = start.strftime("%Y-%m-%d %H:%M:%S")
    end_s = end.strftime("%Y-%m-%d %H:%M:%S")

    imports_sql = f"""
        SELECT valid_time, sum(value_mw) AS imports_mw
        FROM nordspot.silver_crossborder
        WHERE to_zone = '{zone_eic}'
          AND valid_time >= '{start_s}'
          AND valid_time <  '{end_s}'
        GROUP BY valid_time ORDER BY valid_time
    """
    exports_sql = f"""
        SELECT valid_time, sum(value_mw) AS exports_mw
        FROM nordspot.silver_crossborder
        WHERE from_zone = '{zone_eic}'
          AND valid_time >= '{start_s}'
          AND valid_time <  '{end_s}'
        GROUP BY valid_time ORDER BY valid_time
    """
    imp = _ch_query_df(ch_client, imports_sql)
    exp = _ch_query_df(ch_client, exports_sql)

    def _to_series(df: pd.DataFrame, col: str) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
        return df.set_index("valid_time")[col]

    imports = _to_series(imp, "imports_mw")
    exports = _to_series(exp, "exports_mw")
    return (imports.subtract(exports, fill_value=0.0)).rename("net_exchange_mw")


# ── Main feature builder ──────────────────────────────────────────────────────


def build_features(
    zone: ZoneConfig,
    start: datetime,
    end: datetime,
    td=None,
    ch_client=None,
) -> pd.DataFrame:
    """Return a feature matrix for all hours in [start, end).

    Parameters
    ----------
    zone:
        ZoneConfig for the target bidding zone.
    start:
        Window start (UTC-aware).
    end:
        Window end (UTC-aware).
    td:
        Active TimeDBClient.  If None, creates one from env vars.
    ch_client:
        Active clickhouse-connect client.  If None, creates one.

    Returns
    -------
    pd.DataFrame
        Columns: ``valid_time``, ``zone``, ``price`` (target), all features.
        ``valid_time`` is a regular column (not the index) for Feast compatibility.
    """
    if td is None:
        from timedb import TimeDBClient

        td = TimeDBClient()

    if ch_client is None:
        from db.clickhouse import get_client

        ch_client = get_client()

    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    eic = zone.entsoe_eic

    # ── Prices ────────────────────────────────────────────────────────────────
    price = _read_series(td, SERIES["prices_raw"]).reindex(idx).rename("price")
    lag_feats = _build_price_lags(price)
    roll_feats = _build_rolling_stats(price)

    # ── Calendar ──────────────────────────────────────────────────────────────
    cal = _build_calendar(idx)

    # ── Weather ───────────────────────────────────────────────────────────────
    temp = (
        _read_series(td, SERIES["weather_temperature"])
        .reindex(idx)
        .rename("temperature_2m")
    )
    wind = (
        _read_series(td, SERIES["weather_wind_speed"])
        .reindex(idx)
        .rename("wind_speed_10m")
    )
    solar = (
        _read_series(td, SERIES["weather_irradiance"])
        .reindex(idx)
        .rename("solar_radiation")
    )
    weather_interactions = _build_weather_interactions(temp, wind, cal["hour"])

    # ── Generation (Silver / ClickHouse) ─────────────────────────────────────
    gen_df = _read_generation(ch_client, eic, start, end).reindex(idx)

    # ── Load (Silver / ClickHouse) ────────────────────────────────────────────
    load_df = _read_load(ch_client, eic, start, end).reindex(idx)

    # ── Cross-border net exchange (Silver / ClickHouse) ───────────────────────
    net_exchange = _read_net_exchange(ch_client, eic, start, end).reindex(idx)

    # ── Assemble ──────────────────────────────────────────────────────────────
    df = pd.concat(
        [
            price,
            *lag_feats,
            *roll_feats,
            cal,
            temp,
            wind,
            solar,
            *weather_interactions,
            gen_df,
            load_df,
            net_exchange,
        ],
        axis=1,
    )
    df.index.name = "valid_time"
    df = df.reset_index()  # valid_time as column — required by Feast
    df.insert(1, "zone", eic)  # zone column immediately after valid_time
    return df


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from timedb import TimeDBClient

    from config.zone_config import load_zone

    zone = load_zone("SE3")
    td = TimeDBClient()

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - pd.Timedelta(days=30)

    print(
        f"Building features for {zone.name} ({zone.entsoe_eic}) "
        f"{start.date()} → {end.date()} ..."
    )
    df = build_features(zone, start, end, td=td)

    print(f"\nShape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    nan_counts = df.isnull().sum()
    if nan_counts.any():
        print(f"\nNaN counts:\n{nan_counts[nan_counts > 0].to_string()}")
    else:
        print("\nNo NaNs — feature matrix is complete.")
