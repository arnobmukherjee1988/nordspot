"""
Fetch day-ahead electricity prices from ENTSO-E.

Writes raw data to Bronze layer (GCS or local Parquet),
then inserts into ClickHouse for querying.

Usage:
    python -m pipeline.fetch_prices --zone SE3
    python -m pipeline.fetch_prices --zone ALL --start 2020-01-01 --end 2026-06-24
"""

import argparse
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

from config.zone_config import ZoneConfig, load_all_zones, load_zone
from db.schema import SERIES, init_schema
from pipeline.lake_writer import LakeWriter
from pipeline.store import write_series

load_dotenv()


def fetch_prices(zone: ZoneConfig, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch hourly DA prices for a zone. Returns DataFrame with [valid_time, value, zone]."""
    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
    series = client.query_day_ahead_prices(
        zone.entsoe_eic,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )
    df = series.reset_index()
    df.columns = ["valid_time", "value"]
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["zone"] = zone.entsoe_eic
    return df.dropna(subset=["value"])


def sync_prices(zone: ZoneConfig, start: datetime, end: datetime) -> int:
    """Fetch prices -> write to Bronze -> write to ClickHouse. Returns row count."""
    df = fetch_prices(zone, start, end)

    # Bronze layer: one Parquet file per day
    writer = LakeWriter()
    for date, day_df in df.groupby(df["valid_time"].dt.date):
        writer.write(
            day_df.reset_index(drop=True),
            data_type="prices",
            zone=zone.entsoe_eic,
            date=date,
        )

    # Silver layer: insert into ClickHouse
    td = init_schema()
    write_series(
        td, SERIES["prices_raw"], df[["valid_time", "value"]], retention="forever"
    )

    return len(df)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ENTSO-E day-ahead prices")
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
        print(f"Fetching prices: {zone_id} {start_dt.date()} -> {end_dt.date()} ...")
        n = sync_prices(zone_cfg, start_dt, end_dt)
        print(f"  [OK] {n} rows written")
