"""Unit tests for GET /health.

Uses FastAPI's TestClient (backed by httpx) — no server process needed.
Verifies response status, schema fields, and zone coverage.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_is_ok():
    response = client.get("/health")
    assert response.json()["status"] == "ok"


def test_health_covers_all_four_zones():
    response = client.get("/health")
    zones = set(response.json()["zones_covered"])
    assert zones == {"SE1", "SE2", "SE3", "SE4"}


def test_health_api_version_is_v1():
    response = client.get("/health")
    assert response.json()["api_version"] == "v1"


def test_health_has_timestamp():
    response = client.get("/health")
    data = response.json()
    assert "timestamp" in data
    assert data["timestamp"] is not None


def test_health_has_model_version():
    response = client.get("/health")
    assert "model_version" in response.json()


def test_health_content_type_is_json():
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]
