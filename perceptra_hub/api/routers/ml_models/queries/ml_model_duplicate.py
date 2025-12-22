
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


router = APIRouter(
    prefix="/models",
)

@router.post(
    "/{model_id}/duplicate",
    response_model=ModelDetailResponse,
    status_code=status.HTTP_201_CREATED
)
async def duplicate_model(
    model_id: str,
    new_name: Optional[str] = None,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Duplicate an existing model (without versions).
    Creates a copy with same task, framework, config, and tags.
    
    Requires: Organization member
    """
    import uuid
    
    # Get original model
    original = await get_model_by_id(model_id, ctx.organization)
    
    # Generate new name if not provided
    if not new_name:
        new_name = f"{original.name} (Copy)"
        
        # Ensure unique name
        counter = 1
        while await sync_to_async(
            lambda: Model.objects.filter(
                organization=ctx.organization,
                name=new_name,
                is_deleted=False
            ).exists()
        )():
            counter += 1
            new_name = f"{original.name} (Copy {counter})"
    else:
        # Check if name already exists
        if await sync_to_async(
            lambda: Model.objects.filter(
                organization=ctx.organization,
                name=new_name,
                is_deleted=False
            ).exists()
        )():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model '{new_name}' already exists"
            )
    
    # Create duplicate
    duplicate = await sync_to_async(Model.objects.create)(
        model_id=str(uuid.uuid4()),
        name=new_name,
        description=original.description,
        organization=original.organization,
        project=original.project,
        task=original.task,
        framework=original.framework,
        default_config=original.default_config.copy() if original.default_config else {},
        created_by=ctx.user
    )
    
    # Copy tags
    tags = await sync_to_async(lambda: list(original.tags.all()))()
    if tags:
        await sync_to_async(duplicate.tags.set)(tags)
    
    return serialize_model_detail(duplicate)