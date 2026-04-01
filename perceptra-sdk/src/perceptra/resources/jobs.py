from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Any

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


class Jobs(SyncAPIResource):

    def list(self, project_id: str, *, skip: int = 0, limit: int = 100) -> list:
        return self._client._request(
            "GET", f"/jobs/project/{project_id}",
            query={"skip": skip, "limit": limit}, cast_to=list,
        )

    def update(
        self,
        project_id: str,
        job_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        assignee_id: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if name is not None: body["name"] = name
        if description is not None: body["description"] = description
        if status is not None: body["status"] = status
        if assignee_id is not None: body["assignee_id"] = assignee_id
        return self._client._request(
            "PUT", f"/projects/{project_id}/jobs/{job_id}",
            body=body, cast_to=dict,
        )

    def assign_images(self, project_id: str, job_id: str, project_image_ids: List[str]) -> dict:
        return self._client._request(
            "POST", f"/projects/{project_id}/jobs/{job_id}/assign",
            body={"project_image_ids": project_image_ids}, cast_to=dict,
        )

    def delete(self, project_id: str, job_id: str) -> None:
        self._client._request(
            "DELETE", f"/projects/{project_id}/jobs/{job_id}",
            cast_to=dict,
        )


class AsyncJobs(AsyncAPIResource):

    async def list(self, project_id: str, *, skip: int = 0, limit: int = 100) -> list:
        return await self._client._request(
            "GET", f"/jobs/project/{project_id}",
            query={"skip": skip, "limit": limit}, cast_to=list,
        )

    async def update(
        self,
        project_id: str,
        job_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        assignee_id: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if name is not None: body["name"] = name
        if description is not None: body["description"] = description
        if status is not None: body["status"] = status
        if assignee_id is not None: body["assignee_id"] = assignee_id
        return await self._client._request(
            "PUT", f"/projects/{project_id}/jobs/{job_id}",
            body=body, cast_to=dict,
        )

    async def assign_images(self, project_id: str, job_id: str, project_image_ids: List[str]) -> dict:
        return await self._client._request(
            "POST", f"/projects/{project_id}/jobs/{job_id}/assign",
            body={"project_image_ids": project_image_ids}, cast_to=dict,
        )

    async def delete(self, project_id: str, job_id: str) -> None:
        await self._client._request(
            "DELETE", f"/projects/{project_id}/jobs/{job_id}",
            cast_to=dict,
        )
