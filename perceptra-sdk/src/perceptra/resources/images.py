from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Any

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource
from perceptra._files import prepare_file, FileInput
from perceptra.types.image import ImageUploadResponse

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


class Images(SyncAPIResource):

    def upload(
        self,
        file: FileInput,
        *,
        name: Optional[str] = None,
        project_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_of_origin: Optional[str] = None,
        storage_profile_id: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> ImageUploadResponse:
        filename, file_data, content_type = prepare_file(file, name=name)
        data: dict[str, Any] = {}
        if name: data["name"] = name
        if project_id: data["project_id"] = project_id
        if tags: data["tags"] = ",".join(tags)
        if source_of_origin: data["source_of_origin"] = source_of_origin
        if storage_profile_id: data["storage_profile_id"] = storage_profile_id
        if batch_id: data["batch_id"] = batch_id
        return self._client._request(
            "POST", "/images/upload",
            files={"file": (filename, file_data, content_type)},
            data=data,
            cast_to=ImageUploadResponse,
        )

    def list(self, *, skip: int = 0, limit: int = 100) -> list:
        return self._client._request("GET", "/images", query={"skip": skip, "limit": limit}, cast_to=list)

    def bulk_delete(self, image_ids: List[str]) -> dict:
        return self._client._request("POST", "/images/bulk-delete", body={"image_ids": image_ids}, cast_to=dict)


class AsyncImages(AsyncAPIResource):

    async def upload(
        self,
        file: FileInput,
        *,
        name: Optional[str] = None,
        project_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_of_origin: Optional[str] = None,
        storage_profile_id: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> ImageUploadResponse:
        filename, file_data, content_type = prepare_file(file, name=name)
        data: dict[str, Any] = {}
        if name: data["name"] = name
        if project_id: data["project_id"] = project_id
        if tags: data["tags"] = ",".join(tags)
        if source_of_origin: data["source_of_origin"] = source_of_origin
        if storage_profile_id: data["storage_profile_id"] = storage_profile_id
        if batch_id: data["batch_id"] = batch_id
        return await self._client._request(
            "POST", "/images/upload",
            files={"file": (filename, file_data, content_type)},
            data=data,
            cast_to=ImageUploadResponse,
        )

    async def list(self, *, skip: int = 0, limit: int = 100) -> list:
        return await self._client._request("GET", "/images", query={"skip": skip, "limit": limit}, cast_to=list)

    async def bulk_delete(self, image_ids: List[str]) -> dict:
        return await self._client._request("POST", "/images/bulk-delete", body={"image_ids": image_ids}, cast_to=dict)
