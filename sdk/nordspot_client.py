"""NordSpot Python SDK — thin client for the NordSpot forecast API.

Usage
-----
    from sdk import NordSpotClient, NordSpotError

    client = NordSpotClient(
        base_url="http://localhost:8000",
        api_key="your-key-here",
    )

    # 24-hour ahead forecast
    forecast = client.get_forecast(zone="SE3", date="2026-07-02")
    for hour in forecast["forecast"]:
        print(f"Hour {hour['hour']:02d}:  {hour['point']:.2f} EUR/MWh")

    # Health probe
    status = client.health()
    print(status["model_version"])

Errors
------
    NordSpotError is raised for any non-2xx HTTP response.  It carries:
        - status_code (int)
        - detail     (str, from the API's JSON "detail" field when available)
"""

from __future__ import annotations

import datetime
import logging
from typing import Union

import requests

logger = logging.getLogger("nordspot.sdk")


class NordSpotError(Exception):
    """Raised when the NordSpot API returns a non-2xx status code.

    Attributes
    ----------
    status_code : int
        HTTP status code returned by the server.
    detail : str
        Human-readable error description from the API response, or the
        raw response text when the body is not valid JSON.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class NordSpotClient:
    """HTTP client for the NordSpot electricity price forecast API.

    Parameters
    ----------
    base_url : str
        Root URL of the API server, e.g. ``"https://api.nordspot.energy"``.
        Trailing slashes are stripped automatically.
    api_key : str
        API key sent as the ``X-API-Key`` header on every request.
    timeout : int | float
        Request timeout in seconds (default: 30).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: Union[int, float] = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": api_key})

    # ── Public methods ────────────────────────────────────────────────────────

    def get_forecast(
        self,
        zone: str,
        date: Union[str, datetime.date],
    ) -> dict:
        """Request a 24-hour ahead probabilistic price forecast.

        Parameters
        ----------
        zone : str
            Swedish bidding zone: one of ``"SE1"``, ``"SE2"``, ``"SE3"``, ``"SE4"``.
        date : str | datetime.date
            Delivery date.  Must be tomorrow or later.  ``datetime.date``
            objects are serialised to ``YYYY-MM-DD`` automatically.

        Returns
        -------
        dict
            Parsed JSON response matching ``ForecastResponse``:
            ``zone``, ``date``, ``model_version``, ``generated_at``,
            ``forecast`` (list of 24 hourly dicts with ``hour``, ``point``,
            ``q05``, ``q95``).

        Raises
        ------
        NordSpotError
            On any non-2xx HTTP response.
        """
        if isinstance(date, datetime.date):
            date = date.isoformat()

        url = f"{self._base_url}/v1/forecast"
        payload = {"zone": zone, "date": date}
        logger.debug("POST %s  body=%s", url, payload)
        response = self._session.post(url, json=payload, timeout=self._timeout)
        return self._parse(response)

    def health(self) -> dict:
        """Fetch the API health status.

        Returns
        -------
        dict
            Parsed JSON response matching ``HealthResponse``:
            ``status``, ``model_version``, ``trained_at``,
            ``zones_covered``, ``api_version``, ``timestamp``.

        Raises
        ------
        NordSpotError
            On any non-2xx HTTP response.
        """
        url = f"{self._base_url}/health"
        logger.debug("GET %s", url)
        response = self._session.get(url, timeout=self._timeout)
        return self._parse(response)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse(response: requests.Response) -> dict:
        """Return parsed JSON or raise NordSpotError on non-2xx."""
        if response.ok:
            return response.json()
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise NordSpotError(status_code=response.status_code, detail=str(detail))
