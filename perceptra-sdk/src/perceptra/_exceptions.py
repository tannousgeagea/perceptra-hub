"""
Exception hierarchy for the Perceptra SDK.

All exceptions inherit from PerceptraError. HTTP errors map to specific
subclasses of APIError based on status code.
"""
from __future__ import annotations

from typing import Any, Optional


class PerceptraError(Exception):
    """Base exception for all Perceptra SDK errors."""


class APIConnectionError(PerceptraError):
    """Raised when the SDK cannot connect to the API (network, DNS, etc.)."""

    def __init__(self, message: str = "Connection error", *, cause: Optional[Exception] = None) -> None:
        self.cause = cause
        super().__init__(message)


class APITimeoutError(APIConnectionError):
    """Raised when a request times out."""

    def __init__(self, message: str = "Request timed out") -> None:
        super().__init__(message)


class APIError(PerceptraError):
    """
    Base class for all HTTP API errors.

    Attributes:
        status_code: HTTP status code.
        message: Human-readable error message (from response ``detail`` field).
        body: Parsed JSON response body, if available.
        headers: Response headers.
    """

    status_code: int
    message: str
    body: Optional[dict[str, Any]]
    headers: dict[str, str]

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.body = body
        self.headers = headers or {}
        super().__init__(f"[{status_code}] {message}")


class AuthenticationError(APIError):
    """401 — Invalid, expired, or missing API key."""


class PermissionDeniedError(APIError):
    """403 — Insufficient permissions, scope, or IP restriction."""


class NotFoundError(APIError):
    """404 — Resource not found."""


class ConflictError(APIError):
    """409 — Duplicate or conflicting resource."""


class UnprocessableEntityError(APIError):
    """422 — Request validation failed."""


class RateLimitError(APIError):
    """
    429 — Rate limit exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying (from ``Retry-After`` header).
    """

    retry_after: Optional[float]

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 429,
        body: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(message, status_code=status_code, body=body, headers=headers)
        raw = (headers or {}).get("retry-after")
        self.retry_after = float(raw) if raw else None


class InternalServerError(APIError):
    """5xx — Server-side error."""


# ── Mapping ──────────────────────────────────────────────────────────────────

_STATUS_MAP: dict[int, type[APIError]] = {
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
}


def raise_for_status(status_code: int, body: Optional[dict[str, Any]], headers: dict[str, str]) -> None:
    """Raise the appropriate APIError subclass for a non-2xx status code."""
    if 200 <= status_code < 300:
        return

    message = "Unknown error"
    if body:
        message = body.get("detail") or body.get("message") or str(body)

    exc_class = _STATUS_MAP.get(status_code)
    if exc_class is None:
        exc_class = InternalServerError if status_code >= 500 else APIError

    raise exc_class(message, status_code=status_code, body=body, headers=headers)
