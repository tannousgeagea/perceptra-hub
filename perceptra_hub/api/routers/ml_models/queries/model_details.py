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

from api.routers.ml_models.utils import serialize_model_detail
from api.routers.ml_models.schemas import ModelDetailResponse
from asgiref.sync import sync_to_async

router = APIRouter(
    prefix="/models",
)

@sync_to_async
def get_model_by_id(model_id: str, organization) -> Model:
    """Fetch model by ID with organization check"""
    try:
        return Model.objects.select_related(
            'task', 'framework', 'project', 'created_by'
        ).prefetch_related(
            'tags',
            'versions__dataset_version__project',
            'versions__created_by',
            'versions__deployed_by'
        ).get(
            model_id=model_id,
            organization=organization,
            is_deleted=False
        )
    except Model.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

@router.get(
    "/{model_id}",
    response_model=ModelDetailResponse
)
async def get_model(
    model_id: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Get detailed information about a specific model.
    
    Requires: Organization member
    """
    model = await get_model_by_id(model_id, ctx.organization)
    return serialize_model_detail(model)
