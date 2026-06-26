"""Unit tests for db/schema.py — verify DDL strings and structure.

No ClickHouse connection required: we test the DDL strings and
function signatures without executing anything against a database.
"""

import pytest

from db.schema import SERIES, ZONE_TABLE_DDL

# ── SERIES registry ───────────────────────────────────────────────────────────


def test_series_ids_are_unique():
    ids = list(SERIES.values())
    assert len(ids) == len(set(ids)), "Duplicate series IDs detected"


def test_required_series_keys_present():
    required = {
        "prices_raw",
        "weather_temperature",
        "generation_total",
        "generation_wind",
        "generation_solar",
        "load_actual",
    }
    assert required.issubset(SERIES.keys())


# ── Zone-table DDL ────────────────────────────────────────────────────────────


def test_zone_table_ddl_has_three_tables():
    assert set(ZONE_TABLE_DDL.keys()) == {
        "generation_actual",
        "load_actual",
        "crossborder_flows",
    }


@pytest.mark.parametrize(
    "table_name", ["generation_actual", "load_actual", "crossborder_flows"]
)
def test_ddl_uses_merge_tree(table_name):
    assert "MergeTree" in ZONE_TABLE_DDL[table_name]


@pytest.mark.parametrize("table_name", ["generation_actual", "load_actual"])
def test_ddl_has_zone_column(table_name):
    assert "zone" in ZONE_TABLE_DDL[table_name]


def test_crossborder_ddl_has_from_and_to_zone():
    ddl = ZONE_TABLE_DDL["crossborder_flows"]
    assert "from_zone" in ddl
    assert "to_zone" in ddl


@pytest.mark.parametrize(
    "table_name", ["generation_actual", "load_actual", "crossborder_flows"]
)
def test_ddl_is_idempotent(table_name):
    """Every DDL must use CREATE TABLE IF NOT EXISTS."""
    assert "IF NOT EXISTS" in ZONE_TABLE_DDL[table_name]


@pytest.mark.parametrize(
    "table_name", ["generation_actual", "load_actual", "crossborder_flows"]
)
def test_ddl_partitions_by_month(table_name):
    assert "toYYYYMM" in ZONE_TABLE_DDL[table_name]
