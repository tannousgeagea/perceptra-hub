"""
Register images by reference (no upload, no copy) — ingest/services.py.

    POST /api/v1/ingest/images          register / update a batch (idempotent)
    POST /api/v1/ingest/images/remove   withdraw registered images (storage untouched)

Machine clients use an API key with scope `ingest:write` (keys without a
scope list are unrestricted) and write permission; with a project, the key's
user needs edit access to that project.
"""
import logging
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from asgiref.sync import sync_to_async
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import RequestContext, require_permission, require_scope
from projects.context import ProjectContext
from projects.resolution import fetch_project_with_access_check

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest")

MAX_ITEMS = 500


class IngestAnnotation(BaseModel):
    external_id: str = Field(..., max_length=255, description="Stable id in the producing system")
    class_name: str = Field(..., max_length=255)
    bbox_xyxyn: List[float] = Field(..., description="[x1, y1, x2, y2] normalized 0-1")
    polygon_xyn: Optional[List[List[float]]] = None
    source: Literal["prediction", "verified"] = "prediction"
    confidence: Optional[float] = Field(None, ge=0, le=1)
    model_version: Optional[str] = Field(None, max_length=50)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestImage(BaseModel):
    external_id: Optional[str] = Field(None, max_length=255, description="Idempotency key in the producing system")
    storage_profile_id: Optional[str] = Field(None, description="Storage profile UUID")
    storage_profile_name: Optional[str] = Field(None, description="…or its name")
    storage_key: str = Field(..., max_length=500)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    file_size: Optional[int] = Field(None, ge=0)
    checksum: Optional[str] = Field(None, max_length=64)
    name: Optional[str] = Field(None, max_length=255)
    captured_at: Optional[str] = None
    source: Optional[str] = Field(None, max_length=255, description="Origin label (defaults to the API key name)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    mark_as_null: Optional[bool] = Field(None, description="Hard negative: nothing to label in this image")
    review_priority: Optional[int] = None
    annotations: Optional[List[IngestAnnotation]] = Field(
        None, description="The image's full annotation set from this producer (omit to leave annotations as they are)",
    )


class IngestOptionsIn(BaseModel):
    auto_assign_job: bool = False
    batch_id: Optional[str] = None
    create_missing_classes: bool = False
    verify_exists: bool = False


class IngestRequest(BaseModel):
    project_id: Optional[UUID] = None
    options: IngestOptionsIn = Field(default_factory=IngestOptionsIn)
    items: List[IngestImage] = Field(..., min_length=1, max_length=MAX_ITEMS)


class IngestResponse(BaseModel):
    total: int
    created: int
    updated: int
    unchanged: int
    errors: int
    results: List[Dict[str, Any]]


class RemoveRequest(BaseModel):
    project_id: Optional[UUID] = None
    external_ids: List[str] = Field(..., min_length=1, max_length=MAX_ITEMS)


class RemoveResponse(BaseModel):
    results: List[Dict[str, Any]]


async def _project_for(ctx: RequestContext, project_id: Optional[UUID]):
    if project_id is None:
        return None
    project, project_role = await fetch_project_with_access_check(ctx.user, ctx.organization, ctx.role, project_id)
    ProjectContext(org_ctx=ctx, project=project, project_role=project_role).require_edit_permission()
    return project


def _source_label(ctx: RequestContext) -> str:
    key = getattr(ctx, "api_key", None)
    return f"api:{key.name}" if key is not None and getattr(key, "name", None) else "api"


@router.post("/images", response_model=IngestResponse, summary="Register images by reference")
async def ingest_images(
    body: IngestRequest,
    ctx: RequestContext = Depends(require_permission("write")),
    _scope: RequestContext = Depends(require_scope("ingest:write")),
):
    from ingest.services import IngestOptions, register_items

    project = await _project_for(ctx, body.project_id)
    if project is None and any(item.annotations for item in body.items):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="annotations need a project_id")

    items = [item.model_dump(exclude_none=False) for item in body.items]
    results = await sync_to_async(register_items)(
        ctx.organization, ctx.user, items, project=project,
        options=IngestOptions(**body.options.model_dump()), source_label=_source_label(ctx),
    )
    count = lambda s: sum(1 for r in results if r["status"] == s)  # noqa: E731
    return IngestResponse(
        total=len(results), created=count("created"), updated=count("updated"),
        unchanged=count("unchanged"), errors=count("error"), results=results,
    )


@router.post("/images/remove", response_model=RemoveResponse, summary="Withdraw images registered by reference")
async def remove_images(
    body: RemoveRequest,
    ctx: RequestContext = Depends(require_permission("write")),
    _scope: RequestContext = Depends(require_scope("ingest:write")),
):
    from ingest.services import remove_items

    project = await _project_for(ctx, body.project_id)
    results = await sync_to_async(remove_items)(ctx.organization, body.external_ids, project=project)
    return RemoveResponse(results=results)
