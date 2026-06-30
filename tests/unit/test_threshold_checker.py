"""Unit tests for monitoring/threshold_checker.py."""

from __future__ import annotations

import pytest

from monitoring.threshold_checker import check_thresholds, load_thresholds

# -- load_thresholds -------------------------------------------------------


def test_load_thresholds_se3():
    t = load_thresholds("SE3")
    assert "rmae_max" in t
    assert "coverage_min" in t
    assert "coverage_max" in t
    assert "pinball_q50_multiplier" in t
    assert "min_hours" in t
    assert t["rmae_max"] > 0
    assert 0 < t["coverage_min"] < t["coverage_max"] < 1


def test_load_thresholds_unknown_zone():
    with pytest.raises(KeyError, match="XY99"):
        load_thresholds("XY99")


# -- check_thresholds - no alert cases ------------------------------------


def test_no_alerts_when_metrics_healthy():
    metrics = {
        "rmae": 0.95,
        "pinball_q50": 10.0,
        "coverage_rate": 0.91,
        "n_hours": 168,
    }
    alerts = check_thresholds("SE3", metrics, baseline_pinball_q50=10.0)
    assert alerts == []


def test_no_alerts_below_min_hours():
    # Even if everything is terrible, skip alerting with too few data points
    metrics = {
        "rmae": 5.0,
        "pinball_q50": 999.0,
        "coverage_rate": 0.10,
        "n_hours": 10,  # below min_hours=48
    }
    alerts = check_thresholds("SE3", metrics, baseline_pinball_q50=10.0)
    assert alerts == []


def test_no_pinball_alert_when_baseline_missing():
    metrics = {
        "rmae": 0.90,
        "pinball_q50": 999.0,  # huge, but no baseline provided
        "coverage_rate": 0.90,
        "n_hours": 168,
    }
    alerts = check_thresholds("SE3", metrics, baseline_pinball_q50=None)
    # no pinball alert without baseline
    assert not any(a["metric"] == "pinball_q50" for a in alerts)


# -- check_thresholds - alert cases ----------------------------------------


def test_rmae_alert_triggered():
    metrics = {
        "rmae": 1.35,  # exceeds 1.2 threshold
        "pinball_q50": 10.0,
        "coverage_rate": 0.90,
        "n_hours": 168,
    }
    alerts = check_thresholds("SE3", metrics, baseline_pinball_q50=10.0)
    rmae_alerts = [a for a in alerts if a["metric"] == "rmae"]
    assert len(rmae_alerts) == 1
    assert rmae_alerts[0]["value"] == pytest.approx(1.35)
    assert "1.2" in rmae_alerts[0]["reason"]


def test_pinball_alert_triggered():
    # baseline=10, multiplier=1.5 -> threshold=15
    metrics = {
        "rmae": 0.90,
        "pinball_q50": 20.0,  # exceeds 15
        "coverage_rate": 0.90,
        "n_hours": 168,
    }
    alerts = check_thresholds("SE3", metrics, baseline_pinball_q50=10.0)
    pb_alerts = [a for a in alerts if a["metric"] == "pinball_q50"]
    assert len(pb_alerts) == 1
    assert pb_alerts[0]["value"] == pytest.approx(20.0)


def test_coverage_too_low_alert():
    metrics = {
        "rmae": 0.90,
        "pinball_q50": 10.0,
        "coverage_rate": 0.70,  # below 0.80
        "n_hours": 168,
    }
    alerts = check_thresholds("SE3", metrics, baseline_pinball_q50=10.0)
    cov_alerts = [a for a in alerts if a["metric"] == "coverage_rate"]
    assert len(cov_alerts) == 1
    assert "too tight" in cov_alerts[0]["reason"]


def test_coverage_too_high_alert():
    metrics = {
        "rmae": 0.90,
        "pinball_q50": 10.0,
        "coverage_rate": 0.99,  # above 0.97
        "n_hours": 168,
    }
    alerts = check_thresholds("SE3", metrics, baseline_pinball_q50=10.0)
    cov_alerts = [a for a in alerts if a["metric"] == "coverage_rate"]
    assert len(cov_alerts) == 1
    assert "too wide" in cov_alerts[0]["reason"]


def test_multiple_alerts_returned():
    metrics = {
        "rmae": 1.50,
        "pinball_q50": 30.0,
        "coverage_rate": 0.60,
        "n_hours": 168,
    }
    alerts = check_thresholds("SE3", metrics, baseline_pinball_q50=10.0)
    assert len(alerts) == 3
    metric_names = {a["metric"] for a in alerts}
    assert metric_names == {"rmae", "pinball_q50", "coverage_rate"}
