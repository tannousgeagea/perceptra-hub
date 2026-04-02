"""httpx auth flow that injects the X-API-Key header."""
from __future__ import annotations

import httpx


class APIKeyAuth(httpx.Auth):
    """Attach ``X-API-Key`` to every outgoing request."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def auth_flow(self, request: httpx.Request):
        request.headers["X-API-Key"] = self._api_key
        yield request
