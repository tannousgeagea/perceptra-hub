from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource
from perceptra.types.version import Version

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


class Versions(SyncAPIResource):

    def create(
        self,
        project_id: str,
        version_name: str,
        *,
        description: Optional[str] = None,
        export_format: Optional[str] = None,
        storage_profile_id: Optional[str] = None,
    ) -> Version:
        body: dict[str, Any] = {"version_name": version_name}
        if description is not None: body["description"] = description
        if export_format is not None: body["export_format"] = export_format
        if storage_profile_id is not None: body["storage_profile_id"] = storage_profile_id
        return self._client._request(
            "POST", f"/projects/{project_id}/versions",
            body=body, cast_to=Version,
        )

    def list(self, project_id: str) -> list:
        return self._client._request("GET", f"/projects/{project_id}/versions", cast_to=list)

    def retrieve(self, project_id: str, version_id: str) -> Version:
        return self._client._request(
            "GET", f"/projects/{project_id}/versions/{version_id}",
            cast_to=Version,
        )

    def export(self, project_id: str, version_id: str, **kwargs: Any) -> dict:
        return self._client._request(
            "POST", f"/projects/{project_id}/versions/{version_id}/export",
            body=kwargs if kwargs else None, cast_to=dict,
        )


class AsyncVersions(AsyncAPIResource):

    async def create(
        self,
        project_id: str,
        version_name: str,
        *,
        description: Optional[str] = None,
        export_format: Optional[str] = None,
        storage_profile_id: Optional[str] = None,
    ) -> Version:
        body: dict[str, Any] = {"version_name": version_name}
        if description is not None: body["description"] = description
        if export_format is not None: body["export_format"] = export_format
        if storage_profile_id is not None: body["storage_profile_id"] = storage_profile_id
        return await self._client._request(
            "POST", f"/projects/{project_id}/versions",
            body=body, cast_to=Version,
        )

    async def list(self, project_id: str) -> list:
        return await self._client._request("GET", f"/projects/{project_id}/versions", cast_to=list)

    async def retrieve(self, project_id: str, version_id: str) -> Version:
        return await self._client._request(
            "GET", f"/projects/{project_id}/versions/{version_id}",
            cast_to=Version,
        )

    async def export(self, project_id: str, version_id: str, **kwargs: Any) -> dict:
        return await self._client._request(
            "POST", f"/projects/{project_id}/versions/{version_id}/export",
            body=kwargs if kwargs else None, cast_to=dict,
        )
