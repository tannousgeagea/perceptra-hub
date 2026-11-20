"""
API for managing compute profiles - where and how to train models.
"""
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from datetime import datetime

from compute.models import ComputeProfile, ComputeProvider, ComputeFallback
from api.dependencies import get_request_context, RequestContext
from api.routers.compute.schemas import (
    ProviderInstanceInfo,
    ComputeProfileResponse,
    ComputeProfileCreateRequest,
    ComputeProviderResponse,
    ComputeProfileUpdateRequest,
    FallbackProviderRequest,
)
from asgiref.sync import sync_to_async

router = APIRouter(
    prefix="/compute",
)

# ============= Helper Functions =============
@sync_to_async
def get_available_providers():
    """Get all available compute providers"""
    return list(
        ComputeProvider.objects.filter(is_active=True).order_by('name')
    )


@sync_to_async
def get_organization_profiles(organization) -> List[ComputeProfile]:
    """Get all compute profiles for organization"""
    return list(
        ComputeProfile.objects.filter(
            organization=organization
        ).select_related('provider').order_by('-is_default', 'name')
    )


@sync_to_async
def active_job_counts(profile_ids: List[int]) -> Dict[int, int]:
    """Queued or running training jobs per compute profile, in one query."""
    from django.db.models import Count
    from compute.models import TrainingJob

    rows = (
        TrainingJob.objects
        .filter(compute_profile_id__in=profile_ids, training_session__status__in=['queued', 'running'])
        .values('compute_profile_id')
        .annotate(n=Count('id'))
    )
    return {row['compute_profile_id']: row['n'] for row in rows}


@sync_to_async
def create_compute_profile(
    ctx: RequestContext,
    data: ComputeProfileCreateRequest
) -> ComputeProfile:
    """Create new compute profile"""
    import uuid
    from cryptography.fernet import Fernet
    from django.conf import settings
    import json
    
    # Verify provider exists
    try:
        provider = ComputeProvider.objects.get(id=data.provider_id, is_active=True)
    except ComputeProvider.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider {data.provider_id} not found"
        )
    
    # Check if credentials required
    if provider.requires_user_credentials and not data.user_credentials:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider.name}' requires user credentials"
        )
    
    # Encrypt credentials if provided
    encrypted_credentials = {}
    if data.user_credentials:
        # Use Fernet encryption (you should store key in settings)
        cipher = Fernet(settings.COMPUTE_CREDENTIALS_KEY.encode())
        encrypted_data = cipher.encrypt(
            json.dumps(data.user_credentials).encode()
        )
        encrypted_credentials = {'encrypted': encrypted_data.decode()}
    
    # Create profile
    profile = ComputeProfile.objects.create(
        profile_id=str(uuid.uuid4()),
        name=data.name,
        organization=ctx.organization,
        provider=provider,
        default_instance_type=data.default_instance_type,
        strategy=data.strategy,
        max_concurrent_jobs=data.max_concurrent_jobs,
        max_cost_per_hour=data.max_cost_per_hour,
        max_training_hours=data.max_training_hours,
        user_credentials=encrypted_credentials,
        is_default=data.is_default,
        created_by=ctx.user
    )
    
    return profile


@sync_to_async
def get_profile_by_id(profile_id: str, organization) -> ComputeProfile:
    """Get compute profile by ID"""
    try:
        return ComputeProfile.objects.select_related('provider').get(
            profile_id=profile_id,
            organization=organization
        )
    except ComputeProfile.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Compute profile {profile_id} not found"
        )


@sync_to_async
def update_profile(
    profile: ComputeProfile,
    data: ComputeProfileUpdateRequest
) -> ComputeProfile:
    """Update compute profile"""
    from cryptography.fernet import Fernet
    from django.conf import settings
    import json
    
    if data.name is not None:
        profile.name = data.name
    if data.default_instance_type is not None:
        profile.default_instance_type = data.default_instance_type
    if data.strategy is not None:
        profile.strategy = data.strategy
    if data.max_concurrent_jobs is not None:
        profile.max_concurrent_jobs = data.max_concurrent_jobs
    if data.max_cost_per_hour is not None:
        profile.max_cost_per_hour = data.max_cost_per_hour
    if data.max_training_hours is not None:
        profile.max_training_hours = data.max_training_hours
    if data.is_active is not None:
        profile.is_active = data.is_active
    if data.is_default is not None:
        profile.is_default = data.is_default
    
    # Update credentials if provided
    if data.user_credentials is not None:
        cipher = Fernet(settings.COMPUTE_CREDENTIALS_KEY.encode())
        encrypted_data = cipher.encrypt(
            json.dumps(data.user_credentials).encode()
        )
        profile.user_credentials = {'encrypted': encrypted_data.decode()}
    
    profile.save()
    return profile


@sync_to_async
def delete_profile(profile: ComputeProfile):
    """Delete compute profile"""
    # Check if any active training jobs use this profile
    from compute.models import TrainingJob
    active_jobs = TrainingJob.objects.filter(
        compute_profile=profile,
        training_session__status__in=['queued', 'running', 'initializing']
    ).count()
    
    if active_jobs > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete profile with {active_jobs} active training jobs"
        )
    
    profile.delete()


def serialize_provider(provider: ComputeProvider) -> dict:
    """Serialize provider to dict"""
    return {
        "id": provider.id,
        "name": provider.name,
        "provider_type": provider.provider_type,
        "description": provider.description,
        "requires_user_credentials": provider.requires_user_credentials,
        "is_active": provider.is_active,
        "available_instances": provider.available_instances
    }


def serialize_profile(profile: ComputeProfile) -> dict:
    """Serialize profile to dict"""
    return {
        "id": profile.profile_id,
        "name": profile.name,
        "provider": serialize_provider(profile.provider),
        "default_instance_type": profile.default_instance_type,
        "strategy": profile.strategy,
        "max_concurrent_jobs": profile.max_concurrent_jobs,
        "max_cost_per_hour": float(profile.max_cost_per_hour) if profile.max_cost_per_hour else None,
        "max_training_hours": profile.max_training_hours,
        "has_credentials": bool(profile.user_credentials),
        "is_active": profile.is_active,
        "is_default": profile.is_default,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at
    }


# ============= API Endpoints =============

@router.get("/providers", response_model=List[ComputeProviderResponse])
async def list_providers(
    ctx: RequestContext = Depends(get_request_context)
):
    """
    List all available compute providers.
    Shows what options are available for training.
    """
    providers = await get_available_providers()
    return [serialize_provider(p) for p in providers]


@router.get("/profiles", response_model=List[ComputeProfileResponse])
async def list_compute_profiles(
    ctx: RequestContext = Depends(get_request_context)
):
    """
    List all compute profiles for the organization.
    """
    profiles = await get_organization_profiles(ctx.organization)
    return [serialize_profile(p) for p in profiles]


@router.post(
    "/profiles",
    response_model=ComputeProfileResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_profile(
    data: ComputeProfileCreateRequest,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Create a new compute profile.
    
    Defines where and how to run training jobs.
    Requires organization admin permissions.
    """
    # Only admins can create compute profiles
    if not ctx.has_role('admin', 'owner'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only organization admins can create compute profiles"
        )
    
    profile = await create_compute_profile(ctx, data)
    return serialize_profile(profile)


@router.get("/profiles/{profile_id}", response_model=ComputeProfileResponse)
async def get_profile(
    profile_id: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """Get specific compute profile details"""
    profile = await get_profile_by_id(profile_id, ctx.organization)
    return serialize_profile(profile)


@router.patch("/profiles/{profile_id}", response_model=ComputeProfileResponse)
async def update_compute_profile(
    profile_id: str,
    data: ComputeProfileUpdateRequest,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Update compute profile settings.
    Requires organization admin permissions.
    """
    if not ctx.has_role('admin', 'owner'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only organization admins can update compute profiles"
        )
    
    profile = await get_profile_by_id(profile_id, ctx.organization)
    updated_profile = await update_profile(profile, data)
    return serialize_profile(updated_profile)


@router.delete("/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_compute_profile(
    profile_id: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Delete compute profile.
    Cannot delete if active training jobs exist.
    """
    if not ctx.has_role('admin', 'owner'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only organization admins can delete compute profiles"
        )
    
    profile = await get_profile_by_id(profile_id, ctx.organization)
    await delete_profile(profile)


@router.post("/profiles/{profile_id}/fallback")
async def add_fallback_provider(
    profile_id: str,
    data: FallbackProviderRequest,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Add fallback provider to compute profile.
    Fallback providers are used when primary provider is unavailable.
    """
    if not ctx.has_role('admin', 'owner'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only organization admins can modify compute profiles"
        )
    
    profile = await get_profile_by_id(profile_id, ctx.organization)
    
    # Verify provider exists
    try:
        provider = await sync_to_async(ComputeProvider.objects.get)(
            id=data.provider_id,
            is_active=True
        )
    except ComputeProvider.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider {data.provider_id} not found"
        )
    
    # Create fallback
    await sync_to_async(ComputeFallback.objects.create)(
        profile=profile,
        provider=provider,
        priority=data.priority
    )
    
    return {"message": "Fallback provider added successfully"}


def _require_org_admin(ctx: RequestContext, what: str) -> None:
    if not ctx.has_role('admin', 'owner'):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Only organization admins can {what}")


async def _test_provider(provider: ComputeProvider, credentials: Optional[Dict[str, Any]], organization) -> Dict[str, Any]:
    """
    A real connection test, by provider type:
    - on-premise agents: are any of the organization's agents online?
    - platform GPU/CPU: runs on this server, nothing to test (tested=False)
    - cloud providers: call the provider with the credentials
    Returns {tested, valid, message}.
    """
    from compute.adapters import get_adapter_for_provider

    if provider.provider_type == 'on-premise-agent':
        credentials = {'organization_id': str(organization.org_id)}
    elif not provider.requires_user_credentials:
        return {"tested": False, "valid": True,
                "message": "Runs on this platform's own workers; there is no connection to test."}
    elif not credentials:
        return {"tested": False, "valid": False, "message": "Enter the provider's credentials first."}

    try:
        adapter = get_adapter_for_provider(provider)
        result = await sync_to_async(adapter.validate_credentials)(credentials)
    except Exception as e:  # the provider's own error is the useful message
        return {"tested": True, "valid": False, "message": f"Connection failed: {e}"}
    return {"tested": True, "valid": bool(result.get('valid')),
            "message": result.get('message') or ("Connected" if result.get('valid') else "Connection failed")}


class ProviderTestRequest(BaseModel):
    credentials: Dict[str, Any] = Field(default_factory=dict)


@router.post("/providers/{provider_id}/validate")
async def validate_provider_credentials(
    provider_id: int,
    data: ProviderTestRequest,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Test credentials for a provider before saving a profile (org admins).
    Nothing is stored.
    """
    _require_org_admin(ctx, "test compute connections")
    provider = await sync_to_async(ComputeProvider.objects.filter(id=provider_id, is_active=True).first)()
    if provider is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Compute provider not found")
    return await _test_provider(provider, data.credentials, ctx.organization)


@router.get("/profiles/{profile_id}/validate")
async def validate_profile_credentials(
    profile_id: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Test a saved profile's connection with its stored credentials (org admins:
    it decrypts and uses them).
    """
    _require_org_admin(ctx, "test compute connections")
    profile = await get_profile_by_id(profile_id, ctx.organization)
    provider = await sync_to_async(lambda: profile.provider)()

    credentials = None
    if profile.user_credentials:
        from cryptography.fernet import Fernet
        from django.conf import settings
        import json

        try:
            cipher = Fernet(settings.COMPUTE_CREDENTIALS_KEY.encode())
            credentials = json.loads(cipher.decrypt(profile.user_credentials['encrypted'].encode()))
        except Exception:
            return {"tested": False, "valid": False,
                    "message": "The saved credentials can't be read; save them again."}
    return await _test_provider(provider, credentials, ctx.organization)
        
@router.get("/recommendations")
async def get_training_recommendations(
    model_size_mb: Optional[float] = None,
    dataset_size_gb: Optional[float] = None,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Get recommended compute profiles based on model/dataset size.
    Helps users choose the right compute option.
    """
    profiles = await get_organization_profiles(ctx.organization)
    busy = await active_job_counts([p.id for p in profiles if p.is_active])

    recommendations = []
    for profile in profiles:
        if not profile.is_active:
            continue
        
        # Simple recommendation logic
        score = 0
        reason = []
        
        # Platform GPU (free/cheap) - always good for small models
        if profile.provider.provider_type == 'platform-gpu':
            if model_size_mb and model_size_mb < 500:  # <500MB
                score += 30
                reason.append("Good for small models")
            if dataset_size_gb and dataset_size_gb < 10:  # <10GB
                score += 20
                reason.append("Suitable dataset size")
            reason.append("No additional cost")
        
        # Cloud providers - better for large models
        elif profile.provider.provider_type in ['aws-sagemaker', 'gcp-vertex', 'azure-ml']:
            if model_size_mb and model_size_mb > 500:
                score += 30
                reason.append("Better for large models")
            if dataset_size_gb and dataset_size_gb > 10:
                score += 20
                reason.append("Handles large datasets well")
            if profile.user_credentials:
                score += 10
                reason.append("Your cloud resources")
        
        # Check availability
        active_jobs = busy.get(profile.id, 0)

        if active_jobs < profile.max_concurrent_jobs:
            score += 20
            reason.append("Available now")
        else:
            reason.append(f"Queue: ~{30 * (active_jobs - profile.max_concurrent_jobs + 1)} min wait")
        
        recommendations.append({
            "profile_id": profile.profile_id,
            "name": profile.name,
            "provider": profile.provider.name,
            "provider_type": profile.provider.provider_type,
            "instance_type": profile.default_instance_type,
            "score": score,
            "reasons": reason,
            "is_default": profile.is_default,
            "estimated_cost_per_hour": float(profile.max_cost_per_hour) if profile.max_cost_per_hour else None
        })
    
    # Sort by score
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        "recommendations": recommendations,
        "suggestion": recommendations[0] if recommendations else None
    }