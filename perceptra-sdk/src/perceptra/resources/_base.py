"""
Base resource classes for sync and async API resources.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


class SyncAPIResource:
    _client: Perceptra

    def __init__(self, client: Perceptra) -> None:
        self._client = client


class AsyncAPIResource:
    _client: AsyncPerceptra

    def __init__(self, client: AsyncPerceptra) -> None:
        self._client = client
