"""Unit tests for api/auth.py - verify_api_key dependency.

Tests run without a real FastAPI request; they call verify_api_key() directly
after patching os.getenv so no environment variable setup is required.

The dependency signature is:
    verify_api_key(api_key: str | None = Security(_api_key_header)) -> str

FastAPI injects api_key from the X-API-Key header at runtime; in tests we
pass it as a plain argument to isolate the validation logic.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.auth import verify_api_key

# -- Helpers -------------------------------------------------------------------


def _call(api_key, monkeypatch, env_keys="key-abc,key-def"):
    """Set NORDSPOT_API_KEYS and call verify_api_key with the given key."""
    monkeypatch.setenv("NORDSPOT_API_KEYS", env_keys)
    return verify_api_key(api_key=api_key)


# -- Valid key tests -----------------------------------------------------------


def test_valid_key_returns_key(monkeypatch):
    result = _call("key-abc", monkeypatch)
    assert result == "key-abc"


def test_second_valid_key_accepted(monkeypatch):
    result = _call("key-def", monkeypatch)
    assert result == "key-def"


def test_multiple_keys_all_accepted(monkeypatch):
    monkeypatch.setenv("NORDSPOT_API_KEYS", "alpha,beta,gamma")
    for key in ("alpha", "beta", "gamma"):
        assert verify_api_key(api_key=key) == key


# -- Invalid / missing key tests -----------------------------------------------


def test_missing_key_raises_401(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        _call(None, monkeypatch)
    assert exc_info.value.status_code == 401


def test_wrong_key_raises_401(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        _call("not-a-key", monkeypatch)
    assert exc_info.value.status_code == 401


def test_empty_string_key_raises_401(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        _call("", monkeypatch)
    assert exc_info.value.status_code == 401


# -- Environment configuration edge cases -------------------------------------


def test_no_env_var_rejects_all(monkeypatch):
    monkeypatch.delenv("NORDSPOT_API_KEYS", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(api_key="key-abc")
    assert exc_info.value.status_code == 401


def test_blank_env_var_rejects_all(monkeypatch):
    monkeypatch.setenv("NORDSPOT_API_KEYS", "   ")
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(api_key="key-abc")
    assert exc_info.value.status_code == 401


def test_keys_with_whitespace_are_stripped(monkeypatch):
    monkeypatch.setenv("NORDSPOT_API_KEYS", "  key-abc  ,  key-def  ")
    assert verify_api_key(api_key="key-abc") == "key-abc"


def test_401_detail_mentions_api_key(monkeypatch):
    monkeypatch.delenv("NORDSPOT_API_KEYS", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(api_key=None)
    assert "API key" in exc_info.value.detail
