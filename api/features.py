"""Feature retrieval for API inference.

Wraps pipeline.features.build_features() to produce exactly 24 rows for
a given zone and delivery date, using the same feature pipeline as training
so there is no training-serving skew.

Design constraint (see PLATFORM_PLAN.md Epic 5 / pipeline/features.py):
    Weather must come from the Open-Meteo *forecast* archive at training
    time and from the live forecast API at inference time — same signal
    type, different endpoint.  build_features() handles this via the
    TimeDB and ClickHouse clients passed in.

Called by: api/routers/forecast.py (Story 5.4)
"""

from __future__ import annotations

import datetime
import logging
from datetime import timezone

import pandas as pd

logger = logging.getLogger("nordspot.api.features")

# History window:  longest lag = 336h, longest rolling window = 720h
# shifted 24h → effective lookback = 744h.  Add 6h margin → 750h.
_HISTORY_HOURS = 750


def get_inference_features(
    zone_name: str,
    date: datetime.date,
    td=None,
    ch_client=None,
) -> pd.DataFrame:
    """Return a 24-row feature DataFrame for ``zone_name`` / ``date`` inference.

    Calls the same ``build_features()`` pipeline used during training to
    guarantee zero training-serving skew.

    Parameters
    ----------
    zone_name:
        Short zone ID accepted by ``load_zone()`` — e.g. ``"SE3"``.
    date:
        Delivery date in UTC (typically tomorrow's date at forecast request
        time).
    td:
        TimeDB client.  ``None`` → auto-created from environment variables.
    ch_client:
        ClickHouse client.  ``None`` → auto-created from environment
        variables.

    Returns
    -------
    pd.DataFrame
        24 rows — one per delivery hour 00–23 UTC — sorted ascending by
        ``valid_time``.  Columns are identical to the training feature
        matrix produced by ``pipeline.features.build_features()``.

    Raises
    ------
    ValueError
        If the data sources contain no data for ``date`` (e.g. the lake
        has not been backfilled that far), or if the resulting slice has
        a row count other than 24 (missing data or DST transition).
    """
    from config.zone_config import load_zone
    from pipeline.features import build_features

    zone = load_zone(zone_name)

    # Build a wide window: enough history for all lag / rolling features
    # plus the 24 target delivery hours.
    target_start = datetime.datetime(
        date.year, date.month, date.day, tzinfo=timezone.utc
    )
    target_end = target_start + datetime.timedelta(hours=24)
    history_start = target_start - datetime.timedelta(hours=_HISTORY_HOURS)

    logger.info(
        "Fetching features for %s on %s  (history window %s → %s)",
        zone_name,
        date,
        history_start.date(),
        target_end.date(),
    )

    df = build_features(zone, history_start, target_end, td=td, ch_client=ch_client)

    # Slice to the 24 delivery hours of the target date (UTC)
    mask = df["valid_time"].dt.date == date
    rows = df[mask].copy().sort_values("valid_time").reset_index(drop=True)

    if rows.empty:
        raise ValueError(
            f"No feature data available for zone '{zone_name}' on {date}. "
            "Ensure the data lake covers this date (run the backfill pipeline)."
        )

    if len(rows) != 24:
        raise ValueError(
            f"Expected 24 feature rows for {zone_name} on {date}, got {len(rows)}. "
            "Possible cause: missing data or DST hour count mismatch."
        )

    logger.info(
        "Feature matrix ready for %s on %s: %d rows × %d columns",
        zone_name,
        date,
        *rows.shape,
    )
    return rows
