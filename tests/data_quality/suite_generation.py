"""Great Expectations suite for Bronze generation Parquet.

Validates a wide generation DataFrame (one fuel-type column per ENTSO-E
source) before promotion to Silver.

Expected schema
---------------
    valid_time : datetime64[ns, UTC]   — no nulls
    zone       : str                   — ENTSO-E EIC code, no nulls
    <fuel_col> : float64               — each MW column must be ≥ 0
"""

from __future__ import annotations

import great_expectations as ge
import pandas as pd

_MANDATORY_COLS = ("valid_time", "zone")


def validate_generation(df: pd.DataFrame) -> ge.dataset.PandasDataset:
    """Run all generation expectations and return the GE dataset.

    Parameters
    ----------
    df:
        Wide DataFrame with ``valid_time``, ``zone``, and one or more
        fuel-type MW columns.

    Returns
    -------
    ge.dataset.PandasDataset
    """
    gdf = ge.from_pandas(df)

    # Mandatory column presence and no-null checks
    # Guard: GE 0.18 raises KeyError when calling value expectations on missing columns
    for col in _MANDATORY_COLS:
        gdf.expect_column_to_exist(col)
        if col in df.columns:
            gdf.expect_column_values_to_not_be_null(col)

    # At least one fuel column must exist
    fuel_cols = [c for c in df.columns if c not in _MANDATORY_COLS]
    gdf.expect_table_column_count_to_be_between(min_value=len(_MANDATORY_COLS) + 1)

    # All MW columns must be non-negative (generation cannot be negative)
    for col in fuel_cols:
        gdf.expect_column_values_to_be_between(col, min_value=0.0)

    return gdf
