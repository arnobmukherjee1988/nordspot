"""
Fetch weather features from Open-Meteo for a bidding zone.

No API key required - Open-Meteo is free and open.

Variables fetched (hourly):
    temperature_2m      (degC)
    wind_speed_10m      (m/s)
    shortwave_radiation (W/m^2)

Usage:
    python -m pipeline.fetch_weather --zone SE3
    python -m pipeline.fetch_weather --zone ALL --start 2020-01-01 --end 2026-06-24
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import openmeteo_requests
import pandas as pd
import requests_cache
from dotenv import load_dotenv
from retry_requests import retry

from config.zone_config import ZoneConfig, load_all_zones, load_zone
from db.schema import SERIES, init_schema
from pipeline.lake_writer import LakeWriter
from pipeline.store import write_series

load_dotenv()

_VARIABLES = ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]
_SERIES_MAP = {
    "temperature_2m": SERIES["weather_temperature"],
    "wind_speed_10m": SERIES["weather_wind_speed"],
    "shortwave_radiation": SERIES["weather_irradiance"],
}


def _build_client() -> openmeteo_requests.Client:
    cache_path = os.getenv("WEATHER_CACHE_PATH", ".weather_cache.sqlite")
    cache = requests_cache.CachedSession(cache_path, expire_after=3600)
    session = retry(cache, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=session)


def fetch_weather(
    zone: ZoneConfig, start: datetime, end: datetime
) -> dict[str, pd.DataFrame]:
    """Fetch hourly weather for a zone. Returns {variable: DataFrame[valid_time, value]}."""
    client = _build_client()
    params = {
        "latitude": zone.lat,
        "longitude": zone.lon,
        "hourly": _VARIABLES,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    responses = client.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )
    hourly = responses[0].Hourly()
    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=timedelta(seconds=int(hourly.Interval())),
        inclusive="left",
    )
    result = {}
    for i, var in enumerate(_VARIABLES):
        df = pd.DataFrame(
            {"valid_time": times, "value": hourly.Variables(i).ValuesAsNumpy()}
        )
        result[var] = df.dropna(subset=["value"])
    return result


def sync_weather(zone: ZoneConfig, start: datetime, end: datetime) -> dict[str, int]:
    """Fetch weather -> write to Bronze -> write to ClickHouse."""
    dfs = fetch_weather(zone, start, end)

    # Bronze layer: one wide Parquet per day (all variables as columns)
    writer = LakeWriter()
    combined = pd.DataFrame({"valid_time": next(iter(dfs.values()))["valid_time"]})
    for var, df in dfs.items():
        combined[var] = df["value"].values
    combined["zone"] = zone.entsoe_eic
    for date, day_df in combined.groupby(combined["valid_time"].dt.date):
        writer.write(
            day_df.reset_index(drop=True),
            data_type="weather",
            zone=zone.entsoe_eic,
            date=date,
        )

    # Silver layer: insert per-variable series into ClickHouse
    td = init_schema()
    counts = {}
    for var, df in dfs.items():
        write_series(td, _SERIES_MAP[var], df, retention="forever")
        counts[var] = len(df)
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo weather data")
    parser.add_argument("--zone", default="SE3", help="Zone ID (e.g. SE3) or ALL")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_str = args.start or os.getenv("TRAIN_START_DATE", "2023-01-01")
    start_dt = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    if args.end:
        end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    zones = (
        load_all_zones() if args.zone == "ALL" else {args.zone: load_zone(args.zone)}
    )

    for zone_id, zone_cfg in zones.items():
        print(f"Fetching weather: {zone_id} {start_dt.date()} -> {end_dt.date()} ...")
        counts = sync_weather(zone_cfg, start_dt, end_dt)
        for var, n in counts.items():
            print(f"  [OK] {var}: {n} rows")
