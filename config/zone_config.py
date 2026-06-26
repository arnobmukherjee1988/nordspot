"""
Zone configuration loader.

Loads and validates zones.yaml using Pydantic.
All pipeline scripts use this instead of hardcoded zone strings.

Usage:
    from config.zone_config import load_zone, load_all_zones

    zone = load_zone("SE3")
    print(zone.entsoe_eic)   # "10Y1001A1001A46L"
    print(zone.lat)          # 59.33
"""

from pathlib import Path

import yaml
from pydantic import BaseModel

_ZONES_FILE = Path(__file__).parent / "zones.yaml"


class ZoneConfig(BaseModel):
    """Configuration for a single bidding zone."""

    entsoe_eic: str  # ENTSO-E EIC code (used in API calls)
    name: str  # Human-readable city name
    country: str  # ISO 2-letter country code
    lat: float  # Latitude for weather fetch
    lon: float  # Longitude for weather fetch
    timezone: str  # IANA timezone string


def _load_raw() -> dict:
    with open(_ZONES_FILE) as f:
        return yaml.safe_load(f)["zones"]


def load_zone(zone_id: str) -> ZoneConfig:
    """Load and validate config for a single zone (e.g. 'SE3')."""
    raw = _load_raw()
    if zone_id not in raw:
        available = ", ".join(raw.keys())
        raise ValueError(f"Unknown zone '{zone_id}'. Available: {available}")
    return ZoneConfig(**raw[zone_id])


def load_all_zones() -> dict[str, ZoneConfig]:
    """Load config for all zones. Returns {zone_id: ZoneConfig}."""
    return {k: ZoneConfig(**v) for k, v in _load_raw().items()}
