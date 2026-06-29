"""Great Expectations suite for Bronze prices Parquet.

Validates that a prices DataFrame meets the data contract before
promotion to Silver.  Call ``validate_prices(df)`` and inspect
``.success`` on the returned result.

Expected schema
---------------
    valid_time  : datetime64[ns, UTC]   unique hourly timestamps
    value       : float64               EUR/MWh — no nulls, sane range
    zone        : str                   ENTSO-E EIC code
"""

from __future__ import annotations

import great_expectations as ge
import pandas as pd

# Electricity price bounds for NordPool SE zones (EUR/MWh).
# Negative prices occur (excess renewables); hard cap at -500.
# Upper cap at 5 000 EUR/MWh (well above any recorded European spike).
PRICE_MIN = -500.0
PRICE_MAX = 5_000.0


def validate_prices(df: pd.DataFrame) -> ge.dataset.PandasDataset:
    """Run all price expectations and return the GE dataset.

    Parameters
    ----------
    df:
        DataFrame with columns ``valid_time``, ``value``, ``zone``.

    Returns
    -------
    ge.dataset.PandasDataset
        Call ``.validate()`` on the returned object for a full result,
        or inspect individual ``.expect_*`` results already attached.
    """
    gdf = ge.from_pandas(df)

    # Column presence
    gdf.expect_column_to_exist("valid_time")
    gdf.expect_column_to_exist("value")
    gdf.expect_column_to_exist("zone")

    # No nulls in critical columns (guard: GE 0.18 raises KeyError on missing cols)
    if "valid_time" in df.columns:
        gdf.expect_column_values_to_not_be_null("valid_time")
        gdf.expect_column_values_to_be_unique("valid_time")
    if "value" in df.columns:
        gdf.expect_column_values_to_not_be_null("value")
        gdf.expect_column_values_to_be_between(
            "value", min_value=PRICE_MIN, max_value=PRICE_MAX
        )
    if "zone" in df.columns:
        gdf.expect_column_values_to_not_be_null("zone")

    return gdf
