"""
Thin ClickHouse client wrapper.

Reads connection details from environment variables:
    CLICKHOUSE_HOST      (default: localhost)
    CLICKHOUSE_PORT      (default: 8123)
    CLICKHOUSE_USER      (default: nordspot)
    CLICKHOUSE_PASSWORD  (default: nordspot)
    CLICKHOUSE_DB        (default: nordspot)

These are set automatically in Docker Compose via the nordspot_clickhouse service.
For local development outside Docker, copy .env.example → .env and adjust as needed.
"""

import os

import clickhouse_connect
from dotenv import load_dotenv

load_dotenv()


def get_client() -> clickhouse_connect.driver.Client:
    """Return a ClickHouse HTTP client from environment config."""
    return clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "nordspot"),
        password=os.getenv("CLICKHOUSE_PASSWORD", "nordspot"),
        database=os.getenv("CLICKHOUSE_DB", "nordspot"),
    )
