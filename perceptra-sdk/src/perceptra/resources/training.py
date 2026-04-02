from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Iterator, AsyncIterator

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource
from perceptra.types.training import TrainingSession

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


def iter_sse_lines(response) -> Iterator[str]:
    """Yield decoded lines from an SSE stream response."""
    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("data: "):
            yield line[6:]
        elif line:
            yield line


async def aiter_sse_lines(response) -> AsyncIterator[str]:
    """Yield decoded lines from an async SSE stream response."""
    async for line in response.aiter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("data: "):
            yield line[6:]
        elif line:
            yield line


class Training(SyncAPIResource):

    def list(
        self,
        *,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        query: dict = {"limit": limit, "offset": offset}
        if project_id is not None: query["project_id"] = project_id
        if model_id is not None: query["model_id"] = model_id
        if search is not None: query["search"] = search
        return self._client._request("GET", "/training-sessions", query=query, cast_to=list)

    def retrieve(self, session_id: str) -> TrainingSession:
        return self._client._request("GET", f"/training-sessions/{session_id}", cast_to=TrainingSession)

    def stream_logs(self, session_id: str) -> Iterator[str]:
        with self._client._stream("GET", f"/training-sessions/{session_id}/logs") as response:
            yield from iter_sse_lines(response)


class AsyncTraining(AsyncAPIResource):

    async def list(
        self,
        *,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        query: dict = {"limit": limit, "offset": offset}
        if project_id is not None: query["project_id"] = project_id
        if model_id is not None: query["model_id"] = model_id
        if search is not None: query["search"] = search
        return await self._client._request("GET", "/training-sessions", query=query, cast_to=list)

    async def retrieve(self, session_id: str) -> TrainingSession:
        return await self._client._request("GET", f"/training-sessions/{session_id}", cast_to=TrainingSession)

    async def stream_logs(self, session_id: str) -> AsyncIterator[str]:
        async with self._client._stream("GET", f"/training-sessions/{session_id}/logs") as response:
            async for line in aiter_sse_lines(response):
                yield line
