"""Great Expectations suite for Bronze weather Parquet.

Validates a weather DataFrame (one row per valid_time, one column per
Open-Meteo variable) before promotion to Silver.

Expected schema
---------------
    valid_time         : datetime64[ns, UTC]
    temperature_2m     : float64   °C      range −50 … +60
    wind_speed_10m     : float64   m/s     non-negative
    shortwave_radiation: float64   W/m²    non-negative
    zone               : str               ENTSO-E EIC code
"""

from __future__ import annotations

import great_expectations as ge
import pandas as pd

TEMP_MIN = -50.0
TEMP_MAX = 60.0


def validate_weather(df: pd.DataFrame) -> ge.dataset.PandasDataset:
    """Run all weather expectations and return the GE dataset.

    Parameters
    ----------
    df:
        DataFrame with columns ``valid_time``, ``temperature_2m``,
        ``wind_speed_10m``, ``shortwave_radiation``, ``zone``.

    Returns
    -------
    ge.dataset.PandasDataset
    """
    gdf = ge.from_pandas(df)

    # Column presence
    for col in (
        "valid_time",
        "temperature_2m",
        "wind_speed_10m",
        "shortwave_radiation",
        "zone",
    ):
        gdf.expect_column_to_exist(col)

    # No nulls + physical plausibility (guard: GE 0.18 raises KeyError on missing cols)
    if "valid_time" in df.columns:
        gdf.expect_column_values_to_not_be_null("valid_time")
    if "temperature_2m" in df.columns:
        gdf.expect_column_values_to_not_be_null("temperature_2m")
        gdf.expect_column_values_to_be_between(
            "temperature_2m", min_value=TEMP_MIN, max_value=TEMP_MAX
        )
    if "wind_speed_10m" in df.columns:
        gdf.expect_column_values_to_not_be_null("wind_speed_10m")
        gdf.expect_column_values_to_be_between("wind_speed_10m", min_value=0.0)
    if "shortwave_radiation" in df.columns:
        gdf.expect_column_values_to_not_be_null("shortwave_radiation")
        gdf.expect_column_values_to_be_between("shortwave_radiation", min_value=0.0)
    if "zone" in df.columns:
        gdf.expect_column_values_to_not_be_null("zone")

    return gdf
