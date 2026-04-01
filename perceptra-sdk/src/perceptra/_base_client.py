"""
Shared client configuration and request logic.
"""
from __future__ import annotations

import os
from typing import Any, TypeVar, Optional, Type, TYPE_CHECKING

import httpx

from perceptra._constants import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    ENV_API_KEY,
    ENV_BASE_URL,
    USER_AGENT,
)
from perceptra._exceptions import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    raise_for_status,
)
from perceptra._transport._auth import APIKeyAuth
from perceptra._transport._retry import should_retry, sleep_for_retry, async_sleep_for_retry
from perceptra._utils import strip_none

if TYPE_CHECKING:
    from pydantic import BaseModel

T = TypeVar("T")


def _resolve_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get(ENV_API_KEY)
    if not key:
        raise AuthenticationError(
            "API key is required. Pass api_key= or set PERCEPTRA_API_KEY env var.",
            status_code=401,
        )
    if not key.startswith(("ph_", "ise_")):
        raise AuthenticationError(
            f"Invalid API key format. Expected key starting with 'ph_', got '{key[:6]}...'",
            status_code=401,
        )
    return key


def _resolve_base_url(base_url: Optional[str]) -> str:
    url = base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL
    return url.rstrip("/")


def _mask_key(api_key: str) -> str:
    if len(api_key) > 12:
        return api_key[:10] + "..." + "****"
    return "****"


class _BaseClient:
    """Shared configuration for sync and async clients."""

    api_key: str
    base_url: str
    timeout: float
    max_retries: int

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.api_key = _resolve_api_key(api_key)
        self.base_url = _resolve_base_url(base_url)
        self.timeout = timeout
        self.max_retries = max_retries

    def _build_url(self, path: str) -> str:
        return f"{self.base_url}/api/v1{path}"

    def _build_headers(self) -> dict[str, str]:
        return {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(api_key=\"{_mask_key(self.api_key)}\", base_url=\"{self.base_url}\")"


def _parse_response(response: httpx.Response, cast_to: Type[T]) -> T:
    """Parse response JSON into a Pydantic model or return raw dict/list."""
    headers = dict(response.headers)
    try:
        body = response.json()
    except Exception:
        body = None

    raise_for_status(response.status_code, body if isinstance(body, dict) else None, headers)

    if cast_to is dict:
        return body  # type: ignore
    if cast_to is list:
        return body  # type: ignore

    # Pydantic model
    from pydantic import BaseModel
    if isinstance(cast_to, type) and issubclass(cast_to, BaseModel):
        if isinstance(body, list):
            return [cast_to.model_validate(item) for item in body]  # type: ignore
        return cast_to.model_validate(body)  # type: ignore

    return body  # type: ignore
