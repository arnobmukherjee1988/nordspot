"""Feature engineering for SE3 day-ahead price forecasting.

Builds an hourly feature matrix from:
    - Lagged electricity prices   (autoregressive + cross-hour features)
    - Calendar features           (hour, weekday, month, holiday, interactions)
    - Weather features            (temperature, wind, solar irradiance, interactions)

v2 additions over baseline:
    - Price lags extended to 23 h, 25 h (cross-hour), 72 h (3 days), 336 h (2 weeks)
    - Interaction features: hour × month, weekend × hour_sin, temperature × hour
"""

from __future__ import annotations

from datetime import datetime, timezone

import holidays
import numpy as np
import pandas as pd
from timedb import TimeDBClient

from db.schema import SERIES

SE_HOLIDAYS = holidays.Sweden()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_series(td: TimeDBClient, series_id: int) -> pd.Series:
    """Pull a single series from TimeDB → hourly pandas Series (UTC index)."""
    df = td.read(series_ids=[series_id], retention="forever")
    if len(df) == 0:
        return pd.Series(dtype=float, name=series_id)
    pdf = df.to_pandas()
    pdf["valid_time"] = pd.to_datetime(pdf["valid_time"], utc=True)
    return pdf.set_index("valid_time")["value"].sort_index()


def _lag(s: pd.Series, hours: int) -> pd.Series:
    return s.shift(hours).rename(f"{s.name}_lag{hours}h")


def _rolling_mean(s: pd.Series, hours: int) -> pd.Series:
    return s.rolling(hours, min_periods=1).mean().rename(f"{s.name}_roll{hours}h")


# ── Main feature builder ──────────────────────────────────────────────────────

def build_features(
    td: TimeDBClient,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Return a feature matrix for all hours in [start, end).

    Args:
        td:    Active TimeDBClient.
        start: Window start (UTC-aware).
        end:   Window end   (UTC-aware).

    Returns:
        DataFrame indexed by ``valid_time`` (UTC hourly).
        Columns: price (target), all feature columns.
    """
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")

    # ── Prices ────────────────────────────────────────────────────────────────
    price = _read_series(td, SERIES["prices_raw"]).reindex(idx).rename("price")

    # Lag features.  All safe at day-ahead gate closure (noon D-1):
    #   23 h, 25 h  — adjacent-hour prices from yesterday (cross-hour context)
    #   24 h, 48 h  — same hour, 1 and 2 days ago
    #   72 h        — same hour, 3 days ago (captures Mon/Fri patterns)
    #   168 h       — same hour, 1 week ago
    #   336 h       — same hour, 2 weeks ago (fortnightly nuclear outage cycles)
    lags = [23, 24, 25, 48, 72, 168, 336]
    lag_feats = [_lag(price, h) for h in lags]

    # Rolling statistics — always shifted by 24 h to prevent any leakage
    price_lag24 = price.shift(24)
    roll_feats = [
        _rolling_mean(price_lag24.rename("price"), 24).rename("price_roll24h"),
        _rolling_mean(price_lag24.rename("price"), 168).rename("price_roll168h"),
    ]

    # ── Calendar ──────────────────────────────────────────────────────────────
    cal = pd.DataFrame(index=idx)
    cal["hour"]       = idx.hour
    cal["weekday"]    = idx.dayofweek
    cal["month"]      = idx.month
    cal["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    cal["is_holiday"] = idx.normalize().map(
        lambda d: int(d.date() in SE_HOLIDAYS)
    )
    cal["hour_of_week"] = idx.dayofweek * 24 + idx.hour

    # Cyclical encoding
    cal["hour_sin"]    = np.sin(2 * np.pi * cal["hour"]    / 24)
    cal["hour_cos"]    = np.cos(2 * np.pi * cal["hour"]    / 24)
    cal["weekday_sin"] = np.sin(2 * np.pi * cal["weekday"] / 7)
    cal["weekday_cos"] = np.cos(2 * np.pi * cal["weekday"] / 7)
    cal["month_sin"]   = np.sin(2 * np.pi * (cal["month"] - 1) / 12)
    cal["month_cos"]   = np.cos(2 * np.pi * (cal["month"] - 1) / 12)

    # Interaction: hour × month — captures seasonal time-of-day price patterns
    # (e.g. evening peak is longer and higher in winter than summer)
    cal["hour_x_month"] = cal["hour"] * cal["month"]

    # Interaction: weekend × hour_sin — weekend daily pattern differs from weekday
    cal["weekend_x_hour"] = cal["is_weekend"] * cal["hour_sin"]

    # ── Weather ───────────────────────────────────────────────────────────────
    temp  = _read_series(td, SERIES["weather_temperature"]).reindex(idx).rename("temperature")
    wind  = _read_series(td, SERIES["weather_wind_speed"]).reindex(idx).rename("wind_speed")
    solar = _read_series(td, SERIES["weather_irradiance"]).reindex(idx).rename("irradiance")

    # Interaction: temperature × wind (cold + windy → higher heating demand)
    temp_x_wind = (temp * wind).rename("temp_x_wind")

    # Interaction: temperature × hour — heating demand varies by hour of day
    # (morning ramp-up, overnight minimum differ with temperature)
    temp_x_hour = (temp * cal["hour"]).rename("temp_x_hour")

    # ── Assemble ──────────────────────────────────────────────────────────────
    df = pd.concat(
        [
            price,
            *lag_feats,
            *roll_feats,
            cal,
            temp, wind, solar,
            temp_x_wind,
            temp_x_hour,
        ],
        axis=1,
    )
    df.index.name = "valid_time"
    return df


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    td = TimeDBClient()

    end   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - pd.Timedelta(days=30)

    print(f"Building features for {start.date()} -> {end.date()} ...")
    df = build_features(td, start, end)

    print(f"\nShape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    nan_counts = df.isnull().sum()
    print(f"\nNaN counts:")
    print(nan_counts[nan_counts > 0].to_string())
