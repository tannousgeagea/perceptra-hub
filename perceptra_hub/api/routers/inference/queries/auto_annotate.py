"""
Trigger model-based auto-annotation for a set of project images.

POST /api/v1/projects/{project_id}/auto-annotate
"""

import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.tasks.auto_annotate import auto_annotate_images
from common_utils.progress.core import track_progress
from ml_models.models import ModelVersion
from projects.models import Project

router = APIRouter()


class AutoAnnotateRequest(BaseModel):
    model_version_id: str
    image_ids: list[int] = Field(..., min_length=1, max_length=5000)
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class AutoAnnotateResponse(BaseModel):
    task_id: str
    status: str
    queued_images: int


@router.post(
    "/projects/{project_id}/auto-annotate",
    response_model=AutoAnnotateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_auto_annotation(
    project_id: int,
    body: AutoAnnotateRequest,
    x_request_id: Annotated[Optional[str], Header()] = None,
    user=Depends(get_current_user),
):
    """
    Queue model inference on the given images and persist predictions as annotations.

    The task runs asynchronously on the `auto_annotate` worker queue.
    Use X-Request-ID as the task_id to poll progress via /api/v1/progress/{task_id}.
    """
    try:
        Project.objects.get(id=project_id)
    except Project.DoesNotExist:
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        ModelVersion.objects.get(version_id=body.model_version_id)
    except ModelVersion.DoesNotExist:
        raise HTTPException(status_code=404, detail="Model version not found")

    task_id = x_request_id or str(uuid.uuid4())
    track_progress(task_id=task_id, percentage=0, message="Queued", status="pending")

    auto_annotate_images.apply_async(
        kwargs={
            "project_id": project_id,
            "image_ids": body.image_ids,
            "model_version_id": body.model_version_id,
            "confidence_threshold": body.confidence_threshold,
            "task_id": task_id,
        },
        queue="auto_annotate",
        task_id=task_id,
    )

    return AutoAnnotateResponse(
        task_id=task_id,
        status="queued",
        queued_images=len(body.image_ids),
    )
