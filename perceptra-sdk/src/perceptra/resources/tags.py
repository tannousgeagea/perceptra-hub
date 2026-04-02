from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


class Tags(SyncAPIResource):

    def list(self, *, skip: int = 0, limit: int = 100) -> list:
        return self._client._request(
            "GET", "/images/tags",
            query={"skip": skip, "limit": limit}, cast_to=list,
        )

    def create(self, name: str, *, color: Optional[str] = None) -> dict:
        body: dict[str, Any] = {"name": name}
        if color is not None: body["color"] = color
        return self._client._request("POST", "/images/tags", body=body, cast_to=dict)

    def delete(self, tag_id: str) -> None:
        self._client._request("DELETE", f"/images/tags/{tag_id}", cast_to=dict)


class AsyncTags(AsyncAPIResource):

    async def list(self, *, skip: int = 0, limit: int = 100) -> list:
        return await self._client._request(
            "GET", "/images/tags",
            query={"skip": skip, "limit": limit}, cast_to=list,
        )

    async def create(self, name: str, *, color: Optional[str] = None) -> dict:
        body: dict[str, Any] = {"name": name}
        if color is not None: body["color"] = color
        return await self._client._request("POST", "/images/tags", body=body, cast_to=dict)

    async def delete(self, tag_id: str) -> None:
        await self._client._request("DELETE", f"/images/tags/{tag_id}", cast_to=dict)
