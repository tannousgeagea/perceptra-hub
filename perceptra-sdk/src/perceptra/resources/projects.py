from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Any

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource
from perceptra.types.project import Project, ProjectListItem, SplitDatasetResponse
from perceptra.types.image import ProjectImagesResponse

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


class Projects(SyncAPIResource):

    def create(
        self,
        name: str,
        project_type_id: int,
        *,
        description: Optional[str] = None,
        visibility_id: Optional[int] = None,
        thumbnail_url: Optional[str] = None,
        settings: Optional[dict] = None,
        annotation_groups: Optional[list] = None,
    ) -> Project:
        body: dict[str, Any] = {"name": name, "project_type_id": project_type_id}
        if description is not None: body["description"] = description
        if visibility_id is not None: body["visibility_id"] = visibility_id
        if thumbnail_url is not None: body["thumbnail_url"] = thumbnail_url
        if settings is not None: body["settings"] = settings
        if annotation_groups is not None: body["annotation_groups"] = annotation_groups
        return self._client._request("POST", "/projects/add", body=body, cast_to=Project)

    def list(self, *, skip: int = 0, limit: int = 100, is_active: Optional[bool] = None) -> list:
        return self._client._request("GET", "/projects", query={"skip": skip, "limit": limit, "is_active": is_active}, cast_to=list)

    def retrieve(self, project_id: str) -> Project:
        return self._client._request("GET", f"/projects/{project_id}", cast_to=Project)

    def delete(self, project_id: str) -> None:
        self._client._request("DELETE", f"/projects/{project_id}", cast_to=dict)

    def list_images(
        self,
        project_id: str,
        *,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        search: Optional[str] = None,
        split: Optional[str] = None,
        annotated: Optional[bool] = None,
        reviewed: Optional[bool] = None,
    ) -> ProjectImagesResponse:
        return self._client._request(
            "GET", f"/projects/{project_id}/images",
            query={"skip": skip, "limit": limit, "status": status, "search": search, "split": split, "annotated": annotated, "reviewed": reviewed},
            cast_to=ProjectImagesResponse,
        )

    def add_images(
        self,
        project_id: str,
        image_ids: List[str],
        *,
        mode_id: Optional[int] = None,
        priority: int = 0,
        auto_assign_job: bool = True,
    ) -> dict:
        return self._client._request(
            "POST", f"/projects/{project_id}/images",
            body={"image_ids": image_ids, "mode_id": mode_id, "priority": priority, "auto_assign_job": auto_assign_job},
            cast_to=dict,
        )

    def split_dataset(
        self,
        project_id: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> SplitDatasetResponse:
        return self._client._request(
            "POST", f"/projects/{project_id}/images/split",
            body={"train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio},
            cast_to=SplitDatasetResponse,
        )


class AsyncProjects(AsyncAPIResource):

    async def create(self, name: str, project_type_id: int, *, description: Optional[str] = None, visibility_id: Optional[int] = None, thumbnail_url: Optional[str] = None, settings: Optional[dict] = None, annotation_groups: Optional[list] = None) -> Project:
        body: dict[str, Any] = {"name": name, "project_type_id": project_type_id}
        if description is not None: body["description"] = description
        if visibility_id is not None: body["visibility_id"] = visibility_id
        if thumbnail_url is not None: body["thumbnail_url"] = thumbnail_url
        if settings is not None: body["settings"] = settings
        if annotation_groups is not None: body["annotation_groups"] = annotation_groups
        return await self._client._request("POST", "/projects/add", body=body, cast_to=Project)

    async def list(self, *, skip: int = 0, limit: int = 100, is_active: Optional[bool] = None) -> list:
        return await self._client._request("GET", "/projects", query={"skip": skip, "limit": limit, "is_active": is_active}, cast_to=list)

    async def retrieve(self, project_id: str) -> Project:
        return await self._client._request("GET", f"/projects/{project_id}", cast_to=Project)

    async def delete(self, project_id: str) -> None:
        await self._client._request("DELETE", f"/projects/{project_id}", cast_to=dict)

    async def list_images(self, project_id: str, *, skip: int = 0, limit: int = 100, status: Optional[str] = None, search: Optional[str] = None, split: Optional[str] = None, annotated: Optional[bool] = None, reviewed: Optional[bool] = None) -> ProjectImagesResponse:
        return await self._client._request("GET", f"/projects/{project_id}/images", query={"skip": skip, "limit": limit, "status": status, "search": search, "split": split, "annotated": annotated, "reviewed": reviewed}, cast_to=ProjectImagesResponse)

    async def add_images(self, project_id: str, image_ids: List[str], *, mode_id: Optional[int] = None, priority: int = 0, auto_assign_job: bool = True) -> dict:
        return await self._client._request("POST", f"/projects/{project_id}/images", body={"image_ids": image_ids, "mode_id": mode_id, "priority": priority, "auto_assign_job": auto_assign_job}, cast_to=dict)

    async def split_dataset(self, project_id: str, train_ratio: float, val_ratio: float, test_ratio: float) -> SplitDatasetResponse:
        return await self._client._request("POST", f"/projects/{project_id}/images/split", body={"train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio}, cast_to=SplitDatasetResponse)
