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
from api.routers.ml_models.schemas import ModelCreateRequest, ModelDetailResponse
from asgiref.sync import sync_to_async

router = APIRouter(
    prefix="/models",
)

@sync_to_async
def create_model_in_db(
    ctx: RequestContext,
    project: Project,
    data: ModelCreateRequest
) -> Model:
    """Create a new model in database"""
    import uuid
    
    # Verify task and framework exist
    try:
        task = ModelTask.objects.get(name=data.task)
    except ModelTask.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task '{data.task}' not found. Available tasks: {', '.join(ModelTask.objects.values_list('name', flat=True))}"
        )
    
    try:
        framework = ModelFramework.objects.get(name=data.framework)
    except ModelFramework.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Framework '{data.framework}' not found. Available frameworks: {', '.join(ModelFramework.objects.values_list('name', flat=True))}"
        )
    
    # Check if model name already exists in this organization
    if Model.objects.filter(
        organization=ctx.organization,
        name=data.name,
        is_deleted=False
    ).exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{data.name}' already exists in this organization"
        )
        
    # Normalize config keys (camelCase to snake_case)
    normalized_config = {}
    if data.config:
        key_mapping = {
            'batchSize': 'batch_size',
            'learningRate': 'learning_rate',
            'epochs': 'epochs',
            'optimizer': 'optimizer',
            'scheduler': 'scheduler'
        }
        for key, value in data.config.items():
            normalized_key = key_mapping.get(key, key)
            normalized_config[normalized_key] = value
    
    # Create model
    model = Model.objects.create(
        model_id=str(uuid.uuid4()),
        name=data.name,
        description=data.description or "",
        organization=ctx.organization,
        project=project,
        task=task,
        framework=framework,
        default_config=normalized_config,
        created_by=ctx.user
    )
    
    # Add tags if provided
    if data.tags:
        # Get or create tags for this organization
        tag_objects = []
        for tag_name in data.tags:
            tag, _ = ModelTag.objects.get_or_create(
                organization=ctx.organization,
                name=tag_name
            )
            tag_objects.append(tag)
        model.tags.set(tag_objects)
    
    return model

@router.post(
    "/projects/{project_id}/models",
    response_model=ModelDetailResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_model(
    project_id: UUID,
    data: ModelCreateRequest,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """
    Create a new ML model in a project.
    
    Requires: Organization member with project access
    """
    # Check permissions (must be able to edit project)
    project_ctx.require_edit_permission()
    
    # Create model
    model = await create_model_in_db(
        RequestContext(
            user=project_ctx.user,
            organization=project_ctx.organization,
            role=project_ctx.org_role
        ),
        project_ctx.project,
        data
    )
    
    # Serialize and return
    return await serialize_model_detail(model)
