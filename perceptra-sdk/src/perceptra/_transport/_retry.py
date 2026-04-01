"""
Retry logic with exponential backoff for transient errors.
"""
from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_BACKOFF = 8.0


def should_retry(response: httpx.Response) -> bool:
    return response.status_code in RETRYABLE_STATUS_CODES


def backoff_delay(attempt: int, retry_after: str | None = None) -> float:
    """
    Calculate delay for the given attempt number.

    Honors ``Retry-After`` header when present. Otherwise uses
    jittered exponential backoff: ``0.5 * 2^attempt``, capped at 8s.
    """
    if retry_after is not None:
        try:
            return min(float(retry_after), MAX_BACKOFF)
        except ValueError:
            pass

    delay = min(0.5 * (2 ** attempt), MAX_BACKOFF)
    jitter = random.uniform(0, delay * 0.25)  # noqa: S311
    return delay + jitter


def sleep_for_retry(attempt: int, response: httpx.Response) -> None:
    """Block the current thread for the appropriate backoff duration."""
    retry_after = response.headers.get("retry-after")
    time.sleep(backoff_delay(attempt, retry_after))


async def async_sleep_for_retry(attempt: int, response: httpx.Response) -> None:
    """Async sleep for the appropriate backoff duration."""
    import anyio

    retry_after = response.headers.get("retry-after")
    await anyio.sleep(backoff_delay(attempt, retry_after))
