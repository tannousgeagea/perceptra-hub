from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


class Classes(SyncAPIResource):

    def list(self, project_id: str, group_id: str) -> list:
        return self._client._request(
            "GET", f"/classes/projects/{project_id}/groups/{group_id}/classes",
            cast_to=list,
        )

    def create(
        self,
        project_id: str,
        group_id: str,
        name: str,
        color: str,
        *,
        class_id: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {"name": name, "color": color}
        if class_id is not None: body["class_id"] = class_id
        return self._client._request(
            "POST", f"/classes/projects/{project_id}/groups/{group_id}/classes",
            body=body, cast_to=dict,
        )

    def update(
        self,
        project_id: str,
        group_id: str,
        class_id: str,
        *,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if name is not None: body["name"] = name
        if color is not None: body["color"] = color
        return self._client._request(
            "PATCH", f"/classes/projects/{project_id}/groups/{group_id}/classes/{class_id}",
            body=body, cast_to=dict,
        )

    def delete(self, project_id: str, group_id: str, class_id: str) -> None:
        self._client._request(
            "DELETE", f"/classes/projects/{project_id}/groups/{group_id}/classes/{class_id}",
            cast_to=dict,
        )


class AsyncClasses(AsyncAPIResource):

    async def list(self, project_id: str, group_id: str) -> list:
        return await self._client._request(
            "GET", f"/classes/projects/{project_id}/groups/{group_id}/classes",
            cast_to=list,
        )

    async def create(
        self,
        project_id: str,
        group_id: str,
        name: str,
        color: str,
        *,
        class_id: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {"name": name, "color": color}
        if class_id is not None: body["class_id"] = class_id
        return await self._client._request(
            "POST", f"/classes/projects/{project_id}/groups/{group_id}/classes",
            body=body, cast_to=dict,
        )

    async def update(
        self,
        project_id: str,
        group_id: str,
        class_id: str,
        *,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if name is not None: body["name"] = name
        if color is not None: body["color"] = color
        return await self._client._request(
            "PATCH", f"/classes/projects/{project_id}/groups/{group_id}/classes/{class_id}",
            body=body, cast_to=dict,
        )

    async def delete(self, project_id: str, group_id: str, class_id: str) -> None:
        await self._client._request(
            "DELETE", f"/classes/projects/{project_id}/groups/{group_id}/classes/{class_id}",
            cast_to=dict,
        )
