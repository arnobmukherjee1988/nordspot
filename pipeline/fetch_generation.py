"""
Fetch actual hourly generation by fuel type from ENTSO-E.

Writes a wide Parquet (one column per fuel type) to Bronze,
then inserts total / wind / solar series into ClickHouse.

Usage:
    python -m pipeline.fetch_generation --zone SE3
    python -m pipeline.fetch_generation --zone ALL --start 2020-01-01 --end 2026-06-24
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


def fetch_generation(zone: ZoneConfig, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch hourly generation by fuel type.

    Returns a wide DataFrame:
        [valid_time, <fuel_type_col>, ..., zone]
    """
    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
    raw = client.query_generation(
        zone.entsoe_eic,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )
    # entsoe-py may return MultiIndex columns (fuel_type, 'Actual Aggregated')
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs("Actual Aggregated", axis=1, level=1, drop_level=True)
    df = raw.reset_index()
    df = df.rename(columns={df.columns[0]: "valid_time"})
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["zone"] = zone.entsoe_eic
    fuel_cols = [c for c in df.columns if c not in ("valid_time", "zone")]
    return df.dropna(how="all", subset=fuel_cols)


def _wind_total(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if "wind" in c.lower()]
    return df[cols].sum(axis=1) if cols else pd.Series(0.0, index=df.index)


def _solar_total(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if "solar" in c.lower()]
    return df[cols].sum(axis=1) if cols else pd.Series(0.0, index=df.index)


def sync_generation(zone: ZoneConfig, start: datetime, end: datetime) -> int:
    """Fetch generation → Bronze (wide Parquet) → ClickHouse (total / wind / solar)."""
    df = fetch_generation(zone, start, end)

    # Bronze: one wide Parquet per day
    writer = LakeWriter()
    for date, day_df in df.groupby(df["valid_time"].dt.date):
        writer.write(
            day_df.reset_index(drop=True),
            data_type="generation",
            zone=zone.entsoe_eic,
            date=date,
        )

    # Silver: total, wind, solar
    fuel_cols = [c for c in df.columns if c not in ("valid_time", "zone")]
    td = init_schema()

    total_df = pd.DataFrame(
        {"valid_time": df["valid_time"], "value": df[fuel_cols].sum(axis=1)}
    )
    write_series(td, SERIES["generation_total"], total_df, retention="forever")

    wind_df = pd.DataFrame({"valid_time": df["valid_time"], "value": _wind_total(df)})
    write_series(td, SERIES["generation_wind"], wind_df, retention="forever")

    solar_df = pd.DataFrame({"valid_time": df["valid_time"], "value": _solar_total(df)})
    write_series(td, SERIES["generation_solar"], solar_df, retention="forever")

    return len(df)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ENTSO-E actual generation")
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
        print(
            f"Fetching generation: {zone_id} {start_dt.date()} -> {end_dt.date()} ..."
        )
        n = sync_generation(zone_cfg, start_dt, end_dt)
        print(f"  [OK] {n} rows written")
