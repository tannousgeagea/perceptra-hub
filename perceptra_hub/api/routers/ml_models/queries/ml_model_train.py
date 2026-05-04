

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
    import uuid
    
    # Get model
    model = await get_model_by_id(model_id, ctx.organization)
    
    # Verify dataset version exists and belongs to same organization
    try:
        dataset = await sync_to_async(DatasetVersion.objects.select_related('project').get)(
            version_id=data.dataset_version_id,
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
    
    # Build training config, injecting model_size from model defaults if not already present
    training_config = dict(data.config) if data.config else {}
    if 'model_params' not in training_config:
        training_config['model_params'] = {}
    if not training_config['model_params'].get('model_size'):
        default_size = model.default_config.get('model_size', '')
        if default_size:
            training_config['model_params']['model_size'] = default_size

    # Resolve effective model_size for the version record
    effective_model_size = (
        training_config['model_params'].get('model_size')
        or model.default_config.get('model_size', '')
    )

    # Create model version
    version_name = data.version_name or f"v{version_number}"
    model_version = await sync_to_async(ModelVersion.objects.create)(
        version_id=str(uuid.uuid4()),
        model=model,
        version_number=version_number,
        version_name=version_name,
        dataset_version=dataset,
        parent_version=parent_version,
        storage_profile=storage_profile,
        config=training_config,
        model_size=effective_model_size,
        status='queued',
        created_by=ctx.user
    )
    
    # Create training session (use the enriched training_config with model_params.model_size)
    session_id = str(uuid.uuid4())
    training_session = await sync_to_async(TrainingSession.objects.create)(
        session_id=session_id,
        model_version=model_version,
        storage_profile=storage_profile,
        config=training_config,
        status='queued',
        triggered_by=ctx.user
    )
    
    # Submit training via orchestrator — both init and submit must run in the
    # same sync thread because __init__ traverses model_version.model.organization
    # via a lazy ORM relation.
    from training.orchestrator import TrainingOrchestrator

    @sync_to_async
    def run_orchestrator():
        orch = TrainingOrchestrator(model_version)
        return orch.submit_training(
            training_session,
            compute_profile_id=data.compute_profile_id,
            agent_id=data.agent_id,
        )

    training_job = await run_orchestrator()
    
    return {
        "model_version_id": model_version.version_id,
        "version_number": version_number,
        "training_session_id": session_id,
        "task_id": training_job.external_job_id or training_job.job_id,
        "status": "queued",
        "compute_provider": training_job.actual_provider.name,
        "instance_type": training_job.instance_type,
        "message": f"Training queued for {model.name} v{version_number} on {training_job.actual_provider.name}"
    }