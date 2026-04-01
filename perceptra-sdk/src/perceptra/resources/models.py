from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Any

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource
from perceptra.types.model import Model, ModelDetail, TrainingTriggerResponse

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


class Models(SyncAPIResource):

    def create(
        self,
        project_id: str,
        name: str,
        task: str,
        framework: str,
        *,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[dict] = None,
    ) -> Model:
        body: dict[str, Any] = {"name": name, "task": task, "framework": framework}
        if description is not None: body["description"] = description
        if tags is not None: body["tags"] = tags
        if config is not None: body["config"] = config
        return self._client._request(
            "POST", f"/models/projects/{project_id}/models",
            body=body, cast_to=Model,
        )

    def list(self, project_id: str) -> list:
        return self._client._request(
            "GET", f"/models/projects/{project_id}/models",
            cast_to=list,
        )

    def retrieve(self, model_id: str) -> ModelDetail:
        return self._client._request("GET", f"/models/{model_id}", cast_to=ModelDetail)

    def update(
        self,
        model_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[dict] = None,
    ) -> Model:
        body: dict[str, Any] = {}
        if name is not None: body["name"] = name
        if description is not None: body["description"] = description
        if tags is not None: body["tags"] = tags
        if config is not None: body["config"] = config
        return self._client._request("PUT", f"/models/{model_id}", body=body, cast_to=Model)

    def delete(self, model_id: str) -> None:
        self._client._request("DELETE", f"/models/{model_id}", cast_to=dict)

    def train(
        self,
        model_id: str,
        dataset_version_id: str,
        *,
        config: Optional[dict] = None,
        parent_version_id: Optional[str] = None,
        version_name: Optional[str] = None,
        compute_profile_id: Optional[str] = None,
    ) -> TrainingTriggerResponse:
        body: dict[str, Any] = {"dataset_version_id": dataset_version_id}
        if config is not None: body["config"] = config
        if parent_version_id is not None: body["parent_version_id"] = parent_version_id
        if version_name is not None: body["version_name"] = version_name
        if compute_profile_id is not None: body["compute_profile_id"] = compute_profile_id
        return self._client._request(
            "POST", f"/models/{model_id}/train",
            body=body, cast_to=TrainingTriggerResponse,
        )


class AsyncModels(AsyncAPIResource):

    async def create(
        self,
        project_id: str,
        name: str,
        task: str,
        framework: str,
        *,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[dict] = None,
    ) -> Model:
        body: dict[str, Any] = {"name": name, "task": task, "framework": framework}
        if description is not None: body["description"] = description
        if tags is not None: body["tags"] = tags
        if config is not None: body["config"] = config
        return await self._client._request(
            "POST", f"/models/projects/{project_id}/models",
            body=body, cast_to=Model,
        )

    async def list(self, project_id: str) -> list:
        return await self._client._request(
            "GET", f"/models/projects/{project_id}/models",
            cast_to=list,
        )

    async def retrieve(self, model_id: str) -> ModelDetail:
        return await self._client._request("GET", f"/models/{model_id}", cast_to=ModelDetail)

    async def update(
        self,
        model_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[dict] = None,
    ) -> Model:
        body: dict[str, Any] = {}
        if name is not None: body["name"] = name
        if description is not None: body["description"] = description
        if tags is not None: body["tags"] = tags
        if config is not None: body["config"] = config
        return await self._client._request("PUT", f"/models/{model_id}", body=body, cast_to=Model)

    async def delete(self, model_id: str) -> None:
        await self._client._request("DELETE", f"/models/{model_id}", cast_to=dict)

    async def train(
        self,
        model_id: str,
        dataset_version_id: str,
        *,
        config: Optional[dict] = None,
        parent_version_id: Optional[str] = None,
        version_name: Optional[str] = None,
        compute_profile_id: Optional[str] = None,
    ) -> TrainingTriggerResponse:
        body: dict[str, Any] = {"dataset_version_id": dataset_version_id}
        if config is not None: body["config"] = config
        if parent_version_id is not None: body["parent_version_id"] = parent_version_id
        if version_name is not None: body["version_name"] = version_name
        if compute_profile_id is not None: body["compute_profile_id"] = compute_profile_id
        return await self._client._request(
            "POST", f"/models/{model_id}/train",
            body=body, cast_to=TrainingTriggerResponse,
        )
