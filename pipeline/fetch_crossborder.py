"""
Fetch hourly cross-border physical flows from ENTSO-E.

Writes Parquet to Bronze. ClickHouse Silver writes are deferred to
Story 1.6 when the per-pair series IDs are confirmed.

Default zone pairs (Sweden internal + key neighbours):
    SE1 → SE2,  SE2 → SE3,  SE3 → SE4
    SE3 → DK2,  SE3 → DE_LU,  SE3 → FI

Usage:
    python -m pipeline.fetch_crossborder --from SE3 --to SE4
    python -m pipeline.fetch_crossborder --all-pairs --start 2020-01-01
"""

import argparse
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

from config.zone_config import load_zone
from pipeline.lake_writer import LakeWriter

load_dotenv()

# Default pairs expressed as (from_zone_id, to_zone_id).
# Both IDs must exist in config/zones.yaml.
DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("SE1", "SE2"),
    ("SE2", "SE3"),
    ("SE3", "SE4"),
]


def fetch_crossborder(
    from_zone_id: str,
    to_zone_id: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch hourly net physical flow from_zone → to_zone.

    Returns DataFrame with columns [valid_time, value, from_zone, to_zone].
    Positive values = net export from from_zone into to_zone.
    """
    from_zone = load_zone(from_zone_id)
    to_zone = load_zone(to_zone_id)

    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
    series = client.query_crossborder_flows(
        from_zone.entsoe_eic,
        to_zone.entsoe_eic,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )
    df = series.reset_index()
    df.columns = ["valid_time", "value"]
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["from_zone"] = from_zone.entsoe_eic
    df["to_zone"] = to_zone.entsoe_eic
    return df.dropna(subset=["value"])


def sync_crossborder(
    from_zone_id: str,
    to_zone_id: str,
    start: datetime,
    end: datetime,
) -> int:
    """Fetch cross-border flow → Bronze Parquet. Returns row count."""
    df = fetch_crossborder(from_zone_id, to_zone_id, start, end)

    writer = LakeWriter()
    pair_key = f"{from_zone_id}--{to_zone_id}"
    for date, day_df in df.groupby(df["valid_time"].dt.date):
        writer.write(
            day_df.reset_index(drop=True),
            data_type="crossborder",
            zone=pair_key,
            date=date,
        )

    return len(df)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ENTSO-E cross-border flows")
    parser.add_argument("--from", dest="from_zone", default=None, help="From zone ID")
    parser.add_argument("--to", dest="to_zone", default=None, help="To zone ID")
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help=f"Run all default pairs: {DEFAULT_PAIRS}",
    )
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

    if args.all_pairs:
        pairs = DEFAULT_PAIRS
    elif args.from_zone and args.to_zone:
        pairs = [(args.from_zone, args.to_zone)]
    else:
        raise SystemExit("Provide --from / --to, or use --all-pairs.")

    for from_id, to_id in pairs:
        print(
            f"Fetching crossborder: {from_id}→{to_id} "
            f"{start_dt.date()} -> {end_dt.date()} ..."
        )
        n = sync_crossborder(from_id, to_id, start_dt, end_dt)
        print(f"  [OK] {n} rows written")
