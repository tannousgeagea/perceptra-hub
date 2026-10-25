from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Optional

from perceptra.resources._base import SyncAPIResource, AsyncAPIResource

if TYPE_CHECKING:
    from perceptra._client import Perceptra
    from perceptra._async_client import AsyncPerceptra

# Server-side batch limit of POST /ingest/images.
MAX_BATCH = 500


def _body(items: List[dict], project_id: Optional[str], **options: Any) -> dict:
    body: dict[str, Any] = {"items": items, "options": {k: v for k, v in options.items() if v is not None}}
    if project_id:
        body["project_id"] = project_id
    return body


def _batches(items: Iterable[dict], size: int) -> Iterable[List[dict]]:
    batch: List[dict] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class Ingest(SyncAPIResource):
    """
    Register images that already live in a storage Perceptra can read
    (typically an external, read-only storage profile) — no upload, no copy.

    Each item: ``storage_profile_id`` or ``storage_profile_name``,
    ``storage_key``, ``width``, ``height`` and optionally ``external_id``
    (idempotency key), ``metadata``, ``tags``, ``mark_as_null``,
    ``review_priority`` and ``annotations`` (each with ``external_id``,
    ``class_name``, ``bbox_xyxyn``, optional ``polygon_xyn``,
    ``source`` = "prediction" | "verified", ``confidence``, ``metadata``).
    Re-registering is idempotent; annotations a person edited are never
    overwritten (reported as conflicts).
    """

    def register_images(
        self,
        items: List[dict],
        *,
        project_id: Optional[str] = None,
        auto_assign_job: Optional[bool] = None,
        batch_id: Optional[str] = None,
        create_missing_classes: Optional[bool] = None,
        verify_exists: Optional[bool] = None,
        batch_size: int = MAX_BATCH,
    ) -> dict:
        """Register `items` (split into server-sized batches); returns the merged result."""
        merged: dict[str, Any] = {"total": 0, "created": 0, "updated": 0, "unchanged": 0, "errors": 0, "results": []}
        for batch in _batches(items, min(batch_size, MAX_BATCH)):
            result = self._client._request(
                "POST", "/ingest/images",
                body=_body(batch, project_id, auto_assign_job=auto_assign_job, batch_id=batch_id,
                           create_missing_classes=create_missing_classes, verify_exists=verify_exists),
                cast_to=dict,
            )
            for key in ("total", "created", "updated", "unchanged", "errors"):
                merged[key] += result.get(key, 0)
            merged["results"].extend(result.get("results", []))
        return merged

    def remove_images(self, external_ids: List[str], *, project_id: Optional[str] = None) -> dict:
        """Withdraw registered images (their storage is never touched)."""
        results: list = []
        for batch in _batches(external_ids, MAX_BATCH):
            body: dict[str, Any] = {"external_ids": batch}
            if project_id:
                body["project_id"] = project_id
            results.extend(self._client._request("POST", "/ingest/images/remove", body=body, cast_to=dict)
                           .get("results", []))
        return {"results": results}


class AsyncIngest(AsyncAPIResource):

    async def register_images(
        self,
        items: List[dict],
        *,
        project_id: Optional[str] = None,
        auto_assign_job: Optional[bool] = None,
        batch_id: Optional[str] = None,
        create_missing_classes: Optional[bool] = None,
        verify_exists: Optional[bool] = None,
        batch_size: int = MAX_BATCH,
    ) -> dict:
        merged: dict[str, Any] = {"total": 0, "created": 0, "updated": 0, "unchanged": 0, "errors": 0, "results": []}
        for batch in _batches(items, min(batch_size, MAX_BATCH)):
            result = await self._client._request(
                "POST", "/ingest/images",
                body=_body(batch, project_id, auto_assign_job=auto_assign_job, batch_id=batch_id,
                           create_missing_classes=create_missing_classes, verify_exists=verify_exists),
                cast_to=dict,
            )
            for key in ("total", "created", "updated", "unchanged", "errors"):
                merged[key] += result.get(key, 0)
            merged["results"].extend(result.get("results", []))
        return merged

    async def remove_images(self, external_ids: List[str], *, project_id: Optional[str] = None) -> dict:
        results: list = []
        for batch in _batches(external_ids, MAX_BATCH):
            body: dict[str, Any] = {"external_ids": batch}
            if project_id:
                body["project_id"] = project_id
            response = await self._client._request("POST", "/ingest/images/remove", body=body, cast_to=dict)
            results.extend(response.get("results", []))
        return {"results": results}
