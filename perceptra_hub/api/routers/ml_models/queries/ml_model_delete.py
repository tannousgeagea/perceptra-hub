
"""
Improved ML Model API with proper authentication and multi-tenancy.
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from datetime import datetime

from ml_models.models import Model, ModelVersion, ModelTag, ModelFramework, ModelTask
from training.models import TrainingSession
from projects.models import Project, Version as DatasetVersion
from api.dependencies import (
    get_request_context,
    get_project_context,
    RequestContext,
    ProjectContext
)

from api.routers.ml_models.utils import serialize_model_detail, get_model_by_id
from api.routers.ml_models.schemas import ModelCreateRequest, ModelDetailResponse, ModelUpdateRequest
from asgiref.sync import sync_to_async
from projects.dependencies import require_project_permission


router = APIRouter(
    prefix="/models",
)



@sync_to_async
def delete_model_from_db(model: Model, user) -> None:
    """Soft delete model"""
    from django.utils import timezone
    
    # Check if model has any non-deleted versions
    active_versions = model.versions.filter(is_deleted=False).count()
    if active_versions > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete model with {active_versions} active versions. Delete versions first."
        )
    
    model.is_deleted = True
    model.deleted_at = timezone.now()
    model.deleted_by = user
    model.save()


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_model(
    model_id: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Delete model (soft delete).
    Cannot delete if model has active versions.
    
    Requires: the admin role in the model's project (or org admin).
    """
    model = await get_model_by_id(model_id, ctx.organization)
    await sync_to_async(require_project_permission)(ctx, model.project, "manage", "Deleting a model")

    await delete_model_from_db(model, ctx.user)