"""
Server-Sent Events decoder for streaming responses (training logs).
"""
from __future__ import annotations

from typing import Iterator, AsyncIterator

import httpx


def iter_sse_lines(response: httpx.Response) -> Iterator[str]:
    """Yield lines from an SSE streaming response."""
    for line in response.iter_lines():
        if line.startswith("data: "):
            yield line[6:]
        elif line and not line.startswith(":"):
            yield line


async def aiter_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Yield lines from an async SSE streaming response."""
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            yield line[6:]
        elif line and not line.startswith(":"):
            yield line
