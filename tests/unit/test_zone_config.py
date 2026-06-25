"""Unit tests for zone configuration loader."""

import pytest

from config.zone_config import load_all_zones, load_zone


def test_load_known_zone():
    zone = load_zone("SE3")
    assert zone.entsoe_eic == "10Y1001A1001A46L"
    assert zone.lat == 59.33
    assert zone.country == "SE"


def test_load_all_zones_returns_four():
    zones = load_all_zones()
    assert set(zones.keys()) == {"SE1", "SE2", "SE3", "SE4"}


def test_unknown_zone_raises():
    with pytest.raises(ValueError, match="Unknown zone"):
        load_zone("XX9")
