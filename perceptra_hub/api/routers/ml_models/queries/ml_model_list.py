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
    
    result = []
    for model in models:
        latest_version = model.get_latest_version()
        result.append({
            "id": model.model_id,
            "name": model.name,
            "description": model.description,
            "task": model.task.name,
            "framework": model.framework.name,
            "tags": [tag.name for tag in model.tags.all()],
            "version_count": model.versions.filter(is_deleted=False).count(),
            "latest_version_number": latest_version.version_number if latest_version else None,
            "latest_status": latest_version.status if latest_version else None,
            "created_at": model.created_at,
            "updated_at": model.updated_at
        })
    
    return result