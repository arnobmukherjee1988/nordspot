"""Fetch SE3 weather features from Open-Meteo and store in TimeDB.

No API key required — Open-Meteo is free and open.

Variables fetched (hourly):
    temperature_2m          → weather_temperature  (°C)
    wind_speed_10m          → weather_wind_speed   (m/s)
    shortwave_radiation     → weather_irradiance   (W/m²)
"""

import os
from datetime import datetime, timezone

import openmeteo_requests
import pandas as pd
import requests_cache
from dotenv import load_dotenv
from retry_requests import retry

from db.schema import SERIES, init_schema
from pipeline.store import write_series

load_dotenv()

SE3_LAT = float(os.getenv("SE3_LAT", "59.33"))
SE3_LON = float(os.getenv("SE3_LON", "18.07"))

_VARIABLES = ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]
_SERIES_MAP = {
    "temperature_2m":      SERIES["weather_temperature"],
    "wind_speed_10m":      SERIES["weather_wind_speed"],
    "shortwave_radiation": SERIES["weather_irradiance"],
}


def _build_client() -> openmeteo_requests.Client:
    _CACHE_PATH = os.getenv("WEATHER_CACHE_PATH", ".weather_cache.sqlite")
    cache = requests_cache.CachedSession(_CACHE_PATH, expire_after=3600)
    session = retry(cache, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=session)


def fetch_weather(start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """Return one tidy DataFrame per weather variable.

    Args:
        start: Fetch start (UTC-aware).
        end:   Fetch end   (UTC-aware).

    Returns:
        Dict mapping variable name → DataFrame[valid_time, value].
    """
    client = _build_client()

    params = {
        "latitude":        SE3_LAT,
        "longitude":       SE3_LON,
        "hourly":          _VARIABLES,
        "start_date":      start.date().isoformat(),
        "end_date":        end.date().isoformat(),
        "timezone":        "UTC",
        "wind_speed_unit": "ms",
    }

    responses = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    response = responses[0]
    hourly = response.Hourly()

    # Build time index
    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    result = {}
    for i, var in enumerate(_VARIABLES):
        df = pd.DataFrame({
            "valid_time": times,
            "value":      hourly.Variables(i).ValuesAsNumpy(),
        })
        df = df.dropna(subset=["value"])
        result[var] = df

    return result


def sync_weather(start: datetime, end: datetime) -> dict[str, int]:
    """Fetch weather and write all variables to TimeDB."""
    td = init_schema()
    dfs = fetch_weather(start, end)
    counts = {}
    for var, df in dfs.items():
        sid = _SERIES_MAP[var]
        write_series(td, sid, df, retention="forever")
        counts[var] = len(df)
    return counts


if __name__ == "__main__":
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_env = os.getenv("SE3_TRAIN_START", "2023-01-01")
    start = datetime.fromisoformat(start_env).replace(tzinfo=timezone.utc)

    print(f"Fetching weather from {start.date()} -> {end.date()} ...")
    counts = sync_weather(start, end)
    for var, n in counts.items():
        print(f"  [OK] {var}: {n} rows")
