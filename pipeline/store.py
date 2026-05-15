"""Helpers for writing fetched data into TimeDB."""

from datetime import datetime

import pandas as pd
from timedb import TimeDBClient


def write_series(
    td: TimeDBClient,
    series_id: int,
    df: pd.DataFrame,
    *,
    retention: str = "forever",
    knowledge_time: datetime | None = None,
) -> None:
    """Write a tidy DataFrame into TimeDB for a given series_id.

    ``df`` must have columns:
        - ``valid_time``  (datetime, UTC, hourly)
        - ``value``       (float)
        - ``series_id``   will be added / overwritten here

    Args:
        td:             Active TimeDBClient.
        series_id:      Integer series ID from db.schema.SERIES.
        df:             DataFrame with valid_time + value columns.
        retention:      TimeDB retention tier — "short" | "medium" | "long".
        knowledge_time: When the data became known (defaults to now).
    """
    df = df.copy()
    df["series_id"] = series_id
    td.write(df, retention=retention, knowledge_time=knowledge_time)
