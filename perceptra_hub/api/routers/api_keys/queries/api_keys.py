"""
FastAPI routes for API Key management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from typing import Optional, List
from uuid import UUID
import logging
from asgiref.sync import sync_to_async
from api.routers.api_keys.schemas import (
    APIKeyCreate,
    APIKeyUpdate,
    APIKeyRotateRequest,
    APIKeyResponse,
    APIKeyCreateResponse,
)
from api.dependencies import get_request_context, RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api-keys")


def _key_to_response(key) -> APIKeyResponse:
    """Convert an APIKey model instance to an APIKeyResponse."""
    return APIKeyResponse(
        id=key.id,
        api_key_id=key.api_key_id,
        key_prefix=key.key_prefix,
        name=key.name,
        description=key.description,
        scope=key.scope,
        user_id=key.owned_by.id if key.owned_by else None,
        user_username=key.owned_by.username if key.owned_by else None,
        permissions=key.permissions,
        is_active=key.is_active,
        usage_count=key.usage_count,
        last_used_at=key.last_used_at,
        created_at=key.created_at,
        expires_at=key.expires_at,
        rate_limit_per_minute=key.rate_limit_per_minute,
        rate_limit_per_hour=key.rate_limit_per_hour,
        scopes=key.scopes or [],
        allowed_ips=key.allowed_ips or [],
        version=key.version,
        created_by_username=key.created_by.username if key.created_by else '',
    )


# ============= Routes =============

@router.post(
    "",
    response_model=APIKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API Key",
)
async def create_api_key(
    data: APIKeyCreate,
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Create a new API key.

    **IMPORTANT**: The full API key is only returned once during creation.
    Store it securely — it cannot be retrieved later.
    """
    ctx.require_role('admin', 'owner')

    @sync_to_async
    def create_key(ctx, data):
        from api_keys.models import APIKey
        from django.contrib.auth import get_user_model
        from datetime import timedelta
        from django.utils import timezone

        User = get_user_model()

        # Validate user if scope=user
        target_user = None
        if data.scope == 'user':
            if not data.user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="user_id required when scope=user",
                )

            try:
                target_user = User.objects.get(id=data.user_id)
            except User.DoesNotExist:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {data.user_id} not found",
                )

            # Verify user belongs to organization
            if not ctx.organization.organizationmembership_set.filter(user=target_user).exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User must be a member of the organization",
                )

        # Generate key — plaintext is NEVER written to DB
        full_key, key_prefix, hashed_key = APIKey.generate_key()

        expires_at = timezone.now() + timedelta(days=data.expires_in_days)

        api_key = APIKey.objects.create(
            key_prefix=key_prefix,
            hashed_key=hashed_key,
            name=data.name,
            description=data.description,
            organization=ctx.organization,
            scope=data.scope,
            scopes=data.scopes,
            owned_by=target_user,
            created_by=ctx.user,
            permissions=data.permissions,
            expires_at=expires_at,
            rate_limit_per_minute=data.rate_limit_per_minute,
            rate_limit_per_hour=data.rate_limit_per_hour,
            allowed_ips=data.allowed_ips,
        )

        return full_key, api_key

    full_key, api_key = await create_key(ctx, data)

    return APIKeyCreateResponse(
        message="API key created successfully. SAVE THIS KEY - it won't be shown again!",
        api_key=full_key,
        key_info=_key_to_response(api_key),
    )


@router.get(
    "",
    response_model=List[APIKeyResponse],
    summary="List API Keys",
)
async def list_api_keys(
    ctx: RequestContext = Depends(get_request_context),
    is_active: Optional[bool] = Query(None),
    scope: Optional[str] = Query(None, pattern="^(organization|user)$"),
):
    """List all API keys for the organization."""

    @sync_to_async
    def get_keys(org, is_active, scope):
        from api_keys.models import APIKey

        queryset = APIKey.objects.filter(
            organization=org,
        ).select_related('owned_by', 'created_by').order_by('-created_at')

        if is_active is not None:
            queryset = queryset.filter(is_active=is_active)
        if scope:
            queryset = queryset.filter(scope=scope)

        return list(queryset)

    keys = await get_keys(ctx.organization, is_active, scope)
    return [_key_to_response(key) for key in keys]


@router.get(
    "/{key_id}",
    response_model=APIKeyResponse,
    summary="Get API Key Details",
)
async def get_api_key(
    key_id: int,
    ctx: RequestContext = Depends(get_request_context),
):
    """Get details of a specific API key."""

    @sync_to_async
    def get_key(org, key_id):
        from api_keys.models import APIKey

        try:
            return APIKey.objects.select_related('owned_by', 'created_by').get(
                id=key_id,
                organization=org,
            )
        except APIKey.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )

    key = await get_key(ctx.organization, key_id)
    return _key_to_response(key)


@router.patch(
    "/{key_id}",
    response_model=APIKeyResponse,
    summary="Update API Key",
)
async def update_api_key(
    key_id: int,
    data: APIKeyUpdate,
    ctx: RequestContext = Depends(get_request_context),
):
    """Update API key settings (name, permissions, rate limits, scopes, allowed IPs)."""
    ctx.require_role('admin', 'owner')

    @sync_to_async
    def update_key(org, key_id, data):
        from api_keys.models import APIKey

        try:
            key = APIKey.objects.select_related('owned_by', 'created_by').get(
                id=key_id,
                organization=org,
            )
        except APIKey.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )

        update_fields = []

        if data.name is not None:
            key.name = data.name
            update_fields.append('name')
        if data.description is not None:
            key.description = data.description
            update_fields.append('description')
        if data.permissions is not None:
            key.permissions = data.permissions
            update_fields.append('permissions')
        if data.is_active is not None:
            key.is_active = data.is_active
            update_fields.append('is_active')
        if data.rate_limit_per_minute is not None:
            key.rate_limit_per_minute = data.rate_limit_per_minute
            update_fields.append('rate_limit_per_minute')
        if data.rate_limit_per_hour is not None:
            key.rate_limit_per_hour = data.rate_limit_per_hour
            update_fields.append('rate_limit_per_hour')
        if data.scopes is not None:
            key.scopes = data.scopes
            update_fields.append('scopes')
        if data.allowed_ips is not None:
            key.allowed_ips = data.allowed_ips
            update_fields.append('allowed_ips')

        if update_fields:
            key.save(update_fields=update_fields)

        return key

    key = await update_key(ctx.organization, key_id, data)
    return _key_to_response(key)


@router.post(
    "/{key_id}/revoke",
    status_code=status.HTTP_200_OK,
    summary="Revoke API Key",
)
async def revoke_api_key(
    key_id: int,
    ctx: RequestContext = Depends(get_request_context),
):
    """Revoke (deactivate) an API key. Cannot be undone."""
    ctx.require_role('admin', 'owner')

    @sync_to_async
    def revoke_key(org, key_id):
        from api_keys.models import APIKey

        try:
            key = APIKey.objects.get(id=key_id, organization=org)
        except APIKey.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )
        key.revoke()
        return key

    key = await revoke_key(ctx.organization, key_id)
    return {
        "message": "API key revoked successfully",
        "key_id": key.id,
        "key_prefix": key.key_prefix,
        "is_active": key.is_active,
    }


@router.post(
    "/{key_id}/renew",
    response_model=APIKeyResponse,
    summary="Renew API Key Expiration",
)
async def renew_api_key(
    key_id: int,
    days: int = Body(90, ge=1, le=365, embed=True),
    ctx: RequestContext = Depends(get_request_context),
):
    """Extend API key expiration by specified days (default: 90)."""
    ctx.require_role('admin', 'owner')

    @sync_to_async
    def renew_key(org, key_id, days):
        from api_keys.models import APIKey

        try:
            key = APIKey.objects.select_related('owned_by', 'created_by').get(
                id=key_id,
                organization=org,
            )
        except APIKey.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )
        key.renew(days=days)
        return key

    key = await renew_key(ctx.organization, key_id, days)
    return _key_to_response(key)


@router.post(
    "/{key_id}/rotate",
    response_model=APIKeyCreateResponse,
    summary="Rotate API Key",
)
async def rotate_api_key(
    key_id: int,
    data: APIKeyRotateRequest = Body(default=APIKeyRotateRequest()),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Rotate an API key. Creates a new key and schedules the old one for deactivation.

    The old key remains valid for the grace period (default: 24 hours).
    Set grace_period_hours=0 to revoke the old key immediately.
    """
    ctx.require_role('admin', 'owner')

    @sync_to_async
    def do_rotate(org, key_id, grace_period_hours):
        from api_keys.models import APIKey
        from datetime import timedelta
        from django.utils import timezone

        try:
            old_key = APIKey.objects.select_related('owned_by', 'created_by').get(
                id=key_id,
                organization=org,
            )
        except APIKey.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )

        if not old_key.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot rotate an inactive key",
            )

        # Generate new key
        full_key, key_prefix, hashed_key = APIKey.generate_key()

        new_key = APIKey.objects.create(
            key_prefix=key_prefix,
            hashed_key=hashed_key,
            name=old_key.name,
            description=old_key.description,
            organization=old_key.organization,
            scope=old_key.scope,
            scopes=old_key.scopes,
            owned_by=old_key.owned_by,
            created_by=ctx.user,
            permissions=old_key.permissions,
            expires_at=timezone.now() + timedelta(days=old_key.expires_at.day if old_key.expires_at else 90),
            rate_limit_per_minute=old_key.rate_limit_per_minute,
            rate_limit_per_hour=old_key.rate_limit_per_hour,
            allowed_ips=old_key.allowed_ips,
            version=old_key.version + 1,
            rotated_from=old_key,
        )

        # Handle old key
        if grace_period_hours == 0:
            old_key.revoke()
        else:
            old_key.expires_at = timezone.now() + timedelta(hours=grace_period_hours)
            old_key.save(update_fields=['expires_at'])

        return full_key, new_key

    full_key, new_key = await do_rotate(ctx.organization, key_id, data.grace_period_hours)

    return APIKeyCreateResponse(
        message=f"API key rotated successfully. Old key expires in {data.grace_period_hours} hours. SAVE THIS NEW KEY!",
        api_key=full_key,
        key_info=_key_to_response(new_key),
    )


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete API Key",
)
async def delete_api_key(
    key_id: int,
    ctx: RequestContext = Depends(get_request_context),
):
    """Permanently delete an API key. This action cannot be undone."""
    ctx.require_role('admin', 'owner')

    @sync_to_async
    def delete_key(org, key_id):
        from api_keys.models import APIKey

        try:
            key = APIKey.objects.get(id=key_id, organization=org)
        except APIKey.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )
        key.delete()

    await delete_key(ctx.organization, key_id)


@router.get(
    "/{key_id}/usage",
    summary="Get API Key Usage Statistics",
)
async def get_api_key_usage(
    key_id: int,
    ctx: RequestContext = Depends(get_request_context),
    days: int = Query(7, ge=1, le=90, description="Number of days to retrieve"),
):
    """Get usage statistics for an API key."""

    @sync_to_async
    def get_usage_stats(org, key_id, days):
        from api_keys.models import APIKey, APIKeyUsageLog
        from django.utils import timezone
        from datetime import timedelta
        from django.db.models import Count, Avg

        try:
            key = APIKey.objects.get(id=key_id, organization=org)
        except APIKey.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )

        start_date = timezone.now() - timedelta(days=days)
        logs = APIKeyUsageLog.objects.filter(
            api_key=key,
            timestamp__gte=start_date,
        )

        total_requests = logs.count()

        stats_by_endpoint = list(
            logs.values('endpoint', 'method')
            .annotate(count=Count('id'))
            .order_by('-count')[:10]
        )

        stats_by_status = list(
            logs.values('status_code')
            .annotate(count=Count('id'))
            .order_by('status_code')
        )

        avg_response_time = logs.aggregate(
            avg=Avg('response_time_ms'),
        )['avg']

        return {
            "key_id": key.id,
            "key_prefix": key.key_prefix,
            "total_requests": total_requests,
            "period_days": days,
            "top_endpoints": stats_by_endpoint,
            "by_status_code": stats_by_status,
            "avg_response_time_ms": round(avg_response_time, 2) if avg_response_time else None,
        }

    return await get_usage_stats(ctx.organization, key_id, days)
