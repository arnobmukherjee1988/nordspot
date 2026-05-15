"""Fetch SE3 day-ahead electricity prices from ENTSO-E and store in TimeDB.

Requires:
    ENTSOE_API_KEY  in .env
    TIMEDB_CH_URL   in .env
"""

import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

from db.schema import SERIES, init_schema
from pipeline.store import write_series

load_dotenv()

# SE3 bidding zone EIC code
SE3_ZONE = "10Y1001A1001A46L"


def fetch_prices(start: datetime, end: datetime) -> pd.DataFrame:
    """Return hourly DA prices for SE3 as a tidy DataFrame.

    Args:
        start: Fetch start (UTC-aware).
        end:   Fetch end   (UTC-aware).

    Returns:
        DataFrame with columns [valid_time, value].
    """
    api_key = os.environ["ENTSOE_API_KEY"]
    client = EntsoePandasClient(api_key=api_key)

    # entsoe-py returns a pandas Series indexed by UTC timestamps
    series = client.query_day_ahead_prices(
        SE3_ZONE,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
    )

    df = series.reset_index()
    df.columns = ["valid_time", "value"]
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df = df.dropna(subset=["value"])
    return df


def sync_prices(start: datetime, end: datetime) -> int:
    """Fetch prices and write to TimeDB.  Returns number of rows written."""
    td = init_schema()
    df = fetch_prices(start, end)
    write_series(td, SERIES["prices_raw"], df, retention="forever")
    return len(df)


if __name__ == "__main__":
    from datetime import timedelta

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_env = os.getenv("SE3_TRAIN_START", "2023-01-01")
    start = datetime.fromisoformat(start_env).replace(tzinfo=timezone.utc)

    print(f"Fetching prices from {start.date()} -> {end.date()} ...")
    n = sync_prices(start, end)
    print(f"[OK] Wrote {n} hourly price rows to TimeDB")
