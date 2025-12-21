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

from api.routers.ml_models.utils import serialize_model_detail, serialize_models
from api.routers.ml_models.schemas import ModelListResponse
from asgiref.sync import sync_to_async

router = APIRouter(
    prefix="/models",
)

@sync_to_async
def list_models_for_project(project: Project):
    """List all models for a project"""
    return list(
        Model.objects.filter(
            project=project,
            is_deleted=False
        ).select_related(
            'task', 'framework', 'created_by'
        ).prefetch_related(
            'tags', 'versions'
        ).order_by('-created_at')
    )
    
    
@sync_to_async
def get_model_latest_version(model:Model):
    return model.get_latest_version()
    
@router.get(
    "/projects/{project_id}/models",
    response_model=List[ModelListResponse]
)
async def list_project_models(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """
    List all models in a project.
    
    Requires: Organization member with project access
    """
    project_ctx.require_project_access()
    
    models = await list_models_for_project(project_ctx.project)
    
    
    return await serialize_models(models)