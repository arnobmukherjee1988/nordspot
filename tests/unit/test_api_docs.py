"""Unit tests for Story 5.6 - API documentation / OpenAPI spec.

All tests are read-only GET requests against the TestClient.
No authentication, model store, or database setup is needed - the OpenAPI
spec and Swagger UI are served unconditionally.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# -- Swagger UI ----------------------------------------------------------------


def test_docs_returns_200():
    response = client.get("/docs")
    assert response.status_code == 200


def test_docs_is_html():
    response = client.get("/docs")
    assert "text/html" in response.headers["content-type"]


# -- OpenAPI JSON spec ---------------------------------------------------------


def test_openapi_json_returns_200():
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200


def test_openapi_json_is_json():
    response = client.get("/api/v1/openapi.json")
    assert "application/json" in response.headers["content-type"]


def test_openapi_title_is_nordspot():
    spec = client.get("/api/v1/openapi.json").json()
    assert spec["info"]["title"] == "NordSpot"


def test_openapi_forecast_path_present():
    spec = client.get("/api/v1/openapi.json").json()
    assert "/v1/forecast" in spec["paths"]


def test_openapi_health_path_present():
    spec = client.get("/api/v1/openapi.json").json()
    assert "/health" in spec["paths"]


def test_openapi_x_api_key_security_scheme_present():
    spec = client.get("/api/v1/openapi.json").json()
    schemes = spec.get("components", {}).get("securitySchemes", {})
    # FastAPI registers APIKeyHeader schemes under the header name
    assert any(
        "APIKey" in k or "api_key" in k.lower() for k in schemes
    ), f"Expected an API key security scheme; got: {list(schemes.keys())}"


def test_openapi_forecast_request_has_example():
    spec = client.get("/api/v1/openapi.json").json()
    schemas = spec.get("components", {}).get("schemas", {})
    fr = schemas.get("ForecastRequest", {})
    assert "example" in fr, "ForecastRequest schema missing 'example'"


def test_openapi_tags_include_forecast_and_health():
    spec = client.get("/api/v1/openapi.json").json()
    tag_names = {t["name"] for t in spec.get("tags", [])}
    assert "Forecast" in tag_names
    assert "Health" in tag_names
