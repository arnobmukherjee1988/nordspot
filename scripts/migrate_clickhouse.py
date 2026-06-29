"""
Idempotent ClickHouse migration runner.

Creates all zone-aware Silver tables (generation_actual, load_actual,
crossborder_flows). Safe to run multiple times - uses CREATE TABLE IF NOT EXISTS.

Usage:
    python scripts/migrate_clickhouse.py
    python scripts/migrate_clickhouse.py --dry-run   # print DDL, no execute
"""

import argparse

from db.schema import ZONE_TABLE_DDL, create_zone_tables


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NordSpot ClickHouse migrations")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print DDL statements without executing them",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.dry_run:
        print("-- DRY RUN - no changes made --------------------------\n")
        for table_name, ddl in ZONE_TABLE_DDL.items():
            print(f"-- Table: {table_name}")
            print(ddl.strip())
            print()
    else:
        print("Running ClickHouse migrations...")
        tables = create_zone_tables()
        for name in tables:
            print(f"  [OK] {name}")
        print(f"\n{len(tables)} table(s) ready.")
