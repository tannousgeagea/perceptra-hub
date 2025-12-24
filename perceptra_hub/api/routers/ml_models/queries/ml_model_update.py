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

@sync_to_async
def update_model_in_db(
    model: Model,
    data: ModelUpdateRequest
) -> Model:
    """Update model details"""
    if data.name is not None:
        # Check for duplicate name
        if Model.objects.filter(
            organization=model.organization,
            name=data.name,
            is_deleted=False
        ).exclude(id=model.id).exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model '{data.name}' already exists"
            )
        model.name = data.name
    
    if data.description is not None:
        model.description = data.description
    
    if data.config is not None:
        # Normalize config keys
        normalized_config = {}
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
        model.default_config = normalized_config
    
    if data.tags is not None:
        tag_objects = []
        for tag_name in data.tags:
            tag, _ = ModelTag.objects.get_or_create(
                organization=model.organization,
                name=tag_name
            )
            tag_objects.append(tag)
        model.tags.set(tag_objects)
    
    model.save()
    return model

@router.patch(
    "/{model_id}",
    response_model=ModelDetailResponse
)
async def update_model(
    model_id: str,
    data: ModelUpdateRequest,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Update model details (name, description, tags, config).
    
    Requires: Organization member with edit permissions
    """
    model = await get_model_by_id(model_id, ctx.organization)
    
    # Check permissions - must be creator or org admin
    if not ctx.has_role('admin', 'owner'):
        if model.created_by and model.created_by.id != ctx.user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only model creator or organization admin can update"
            )
    
    updated_model = await update_model_in_db(model, data)
    return await serialize_model_detail(updated_model)
