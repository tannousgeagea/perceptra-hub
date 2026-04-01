from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Any

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource
from perceptra.types.annotation import Annotation, AnnotationCreateResponse

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra


class Annotations(SyncAPIResource):

    def create(
        self,
        project_id: str,
        image_id: str,
        annotation_type: str,
        data: List[float],
        *,
        annotation_class_id: Optional[int] = None,
        annotation_class_name: Optional[str] = None,
        annotation_uid: Optional[str] = None,
        annotation_source: Optional[str] = None,
        confidence: Optional[float] = None,
        annotation_time_seconds: Optional[float] = None,
    ) -> AnnotationCreateResponse:
        body: dict[str, Any] = {
            "annotation_type": annotation_type,
            "data": data,
        }
        if annotation_class_id is not None: body["annotation_class_id"] = annotation_class_id
        if annotation_class_name is not None: body["annotation_class_name"] = annotation_class_name
        if annotation_uid is not None: body["annotation_uid"] = annotation_uid
        if annotation_source is not None: body["annotation_source"] = annotation_source
        if confidence is not None: body["confidence"] = confidence
        if annotation_time_seconds is not None: body["annotation_time_seconds"] = annotation_time_seconds
        return self._client._request(
            "POST", f"/projects/{project_id}/images/{image_id}/annotations",
            body=body, cast_to=AnnotationCreateResponse,
        )

    def batch_create(self, project_id: str, items: List[dict]) -> dict:
        return self._client._request(
            "POST", f"/projects/{project_id}/annotations/batch",
            body={"items": items}, cast_to=dict,
        )

    def list(self, project_id: str, image_id: str) -> list:
        return self._client._request(
            "GET", f"/projects/{project_id}/images/{image_id}/annotations",
            cast_to=list,
        )

    def update(
        self,
        project_id: str,
        image_id: str,
        annotation_id: str,
        *,
        annotation_class_id: Optional[int] = None,
        data: Optional[List[float]] = None,
        reviewed: Optional[bool] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if annotation_class_id is not None: body["annotation_class_id"] = annotation_class_id
        if data is not None: body["data"] = data
        if reviewed is not None: body["reviewed"] = reviewed
        return self._client._request(
            "PATCH", f"/projects/{project_id}/images/{image_id}/annotations/{annotation_id}",
            body=body, cast_to=dict,
        )

    def delete(self, project_id: str, image_id: str, annotation_id: str) -> None:
        self._client._request(
            "DELETE", f"/projects/{project_id}/images/{image_id}/annotations/{annotation_id}",
            cast_to=dict,
        )


class AsyncAnnotations(AsyncAPIResource):

    async def create(
        self,
        project_id: str,
        image_id: str,
        annotation_type: str,
        data: List[float],
        *,
        annotation_class_id: Optional[int] = None,
        annotation_class_name: Optional[str] = None,
        annotation_uid: Optional[str] = None,
        annotation_source: Optional[str] = None,
        confidence: Optional[float] = None,
        annotation_time_seconds: Optional[float] = None,
    ) -> AnnotationCreateResponse:
        body: dict[str, Any] = {
            "annotation_type": annotation_type,
            "data": data,
        }
        if annotation_class_id is not None: body["annotation_class_id"] = annotation_class_id
        if annotation_class_name is not None: body["annotation_class_name"] = annotation_class_name
        if annotation_uid is not None: body["annotation_uid"] = annotation_uid
        if annotation_source is not None: body["annotation_source"] = annotation_source
        if confidence is not None: body["confidence"] = confidence
        if annotation_time_seconds is not None: body["annotation_time_seconds"] = annotation_time_seconds
        return await self._client._request(
            "POST", f"/projects/{project_id}/images/{image_id}/annotations",
            body=body, cast_to=AnnotationCreateResponse,
        )

    async def batch_create(self, project_id: str, items: List[dict]) -> dict:
        return await self._client._request(
            "POST", f"/projects/{project_id}/annotations/batch",
            body={"items": items}, cast_to=dict,
        )

    async def list(self, project_id: str, image_id: str) -> list:
        return await self._client._request(
            "GET", f"/projects/{project_id}/images/{image_id}/annotations",
            cast_to=list,
        )

    async def update(
        self,
        project_id: str,
        image_id: str,
        annotation_id: str,
        *,
        annotation_class_id: Optional[int] = None,
        data: Optional[List[float]] = None,
        reviewed: Optional[bool] = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if annotation_class_id is not None: body["annotation_class_id"] = annotation_class_id
        if data is not None: body["data"] = data
        if reviewed is not None: body["reviewed"] = reviewed
        return await self._client._request(
            "PATCH", f"/projects/{project_id}/images/{image_id}/annotations/{annotation_id}",
            body=body, cast_to=dict,
        )

    async def delete(self, project_id: str, image_id: str, annotation_id: str) -> None:
        await self._client._request(
            "DELETE", f"/projects/{project_id}/images/{image_id}/annotations/{annotation_id}",
            cast_to=dict,
        )
