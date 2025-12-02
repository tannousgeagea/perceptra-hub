

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
from api.routers.ml_models.schemas import TrainingTriggerRequest, TrainingTriggerResponse
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

@router.post(
    "/{model_id}/train",
    response_model=TrainingTriggerResponse
)
async def trigger_training(
    model_id: str,
    data: TrainingTriggerRequest,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Trigger training for a new version of a model.
    
    Requires: Organization member with edit permissions
    """
    from event_api.tasks.train_model.core import train_model
    import uuid
    
    # Get model
    model = await get_model_by_id(model_id, ctx.organization)
    
    # Verify dataset version exists and belongs to same organization
    try:
        dataset = await sync_to_async(DatasetVersion.objects.select_related('project').get)(
            id=data.dataset_version_id,
            project__organization=ctx.organization
        )
    except DatasetVersion.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset version {data.dataset_version_id} not found"
        )
    
    # Get parent version if specified
    parent_version = None
    if data.parent_version_id:
        try:
            parent_version = await sync_to_async(ModelVersion.objects.get)(
                version_id=data.parent_version_id,
                model=model,
                is_deleted=False
            )
        except ModelVersion.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Parent version {data.parent_version_id} not found"
            )
    
    # Calculate next version number
    last_version = await sync_to_async(
        lambda: ModelVersion.objects.filter(model=model).order_by('-version_number').first()
    )()
    version_number = (last_version.version_number + 1) if last_version else 1
    
    # Get storage profile (use default or first active)
    storage_profile = await sync_to_async(
        lambda: ctx.organization.storage_profiles.filter(
            is_default=True,
            is_active=True
        ).first()
    )()
    
    if not storage_profile:
        # Fallback to first active profile
        storage_profile = await sync_to_async(
            lambda: ctx.organization.storage_profiles.filter(is_active=True).first()
        )()
    
    if not storage_profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No default storage profile configured for organization"
        )
    
    # Create model version
    model_version = await sync_to_async(ModelVersion.objects.create)(
        version_id=str(uuid.uuid4()),
        model=model,
        version_number=version_number,
        version_name=data.version_name,
        dataset_version=dataset,
        parent_version=parent_version,
        storage_profile=storage_profile,
        config=data.config,
        status='queued',
        created_by=ctx.user
    )
    
    # Create training session
    session_id = str(uuid.uuid4())
    training_session = await sync_to_async(TrainingSession.objects.create)(
        session_id=session_id,
        model_version=model_version,
        storage_profile=storage_profile,
        config=data.config,
        status='queued',
        triggered_by=ctx.user
    )
    
    # Trigger async training task
    task = train_model.apply_async(
        args=(model_version.version_id, parent_version.version_id if parent_version else None),
        task_id=session_id
    )
    
    # Update training session with task ID
    training_session.task_id = task.id
    await sync_to_async(training_session.save)()
    
    return {
        "model_version_id": model_version.version_id,
        "version_number": version_number,
        "training_session_id": session_id,
        "task_id": task.id,
        "status": "queued",
        "message": f"Training queued for {model.name} v{version_number}"
    }