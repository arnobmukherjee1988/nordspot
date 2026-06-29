"""API key authentication dependency for NordSpot.

Usage
-----
Add ``Depends(verify_api_key)`` to any endpoint that should require auth:

    @router.post("/forecast", dependencies=[Depends(verify_api_key)])
    async def forecast(...): ...

Configuration
-------------
Set the ``NORDSPOT_API_KEYS`` environment variable to a comma-separated list
of valid keys before starting the API:

    NORDSPOT_API_KEYS=key-abc123,key-def456 uvicorn api.main:app

If the variable is unset or empty, ALL requests are rejected with 401 — the
API refuses to serve data with no keys configured.
"""

from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger("nordspot.api.auth")

# FastAPI security scheme — reads X-API-Key from request headers.
# auto_error=False lets us return a clean 401 rather than FastAPI's default 403.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _load_valid_keys() -> frozenset[str]:
    """Read NORDSPOT_API_KEYS env var and return the set of valid keys.

    Called once per request (cheap — just os.getenv + split).
    Returns an empty frozenset when the variable is unset or blank,
    which causes every request to be rejected.
    """
    raw = os.getenv("NORDSPOT_API_KEYS", "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    return frozenset(keys)


def verify_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """FastAPI dependency that enforces API key authentication.

    Parameters
    ----------
    api_key:
        Value of the ``X-API-Key`` request header, injected by FastAPI.
        ``None`` when the header is absent.

    Returns
    -------
    str
        The validated API key (useful for per-key logging in callers).

    Raises
    ------
    HTTPException(401)
        When the header is missing or the key is not in the allowed set.
    """
    valid_keys = _load_valid_keys()

    if not api_key or api_key not in valid_keys:
        logger.warning("Rejected request — invalid or missing X-API-Key")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Include a valid X-API-Key header.",
        )

    return api_key
