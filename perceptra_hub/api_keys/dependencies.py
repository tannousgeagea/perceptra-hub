"""
FastAPI dependencies for API Key authentication and rate limiting.
Add these to your existing api/dependencies.py
"""
from fastapi import Header, HTTPException, status, Request
from typing import Optional, Tuple
from datetime import datetime, timedelta
from django.utils import timezone
from asgiref.sync import sync_to_async
from django.contrib.auth import get_user_model
from organizations.models import Organization
from api_keys.models import APIKey
import logging

logger = logging.getLogger(__name__)
User = get_user_model()


# ============= API Key Authentication =============

@sync_to_async
def fetch_user_and_org_from_api_key(api_key: str) -> Tuple[User, Organization, 'APIKey']:
    """
    Fetch user, organization, and API key object from API key.
    Validates key and updates usage tracking.
    """
    from api_keys.models import APIKey
    
    # Hash the incoming key for comparison
    hashed_key = APIKey.hash_key(api_key)
    
    try:
        key_obj = APIKey.objects.select_related(
            'organization', 'user', 'created_by'
        ).get(hashed_key=hashed_key)
    except APIKey.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Validate key
    if not key_obj.is_valid():
        detail = "API key is inactive"
        if key_obj.expires_at and key_obj.expires_at < timezone.now():
            detail = f"API key expired on {key_obj.expires_at.strftime('%Y-%m-%d')}"
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail
        )
    
    # Increment usage (async, non-blocking)
    key_obj.increment_usage()
    
    # Determine user based on scope
    # For org-wide keys, use created_by as the acting user
    # For user-specific keys, use the specified user
    user = key_obj.user if key_obj.scope == 'user' else key_obj.created_by
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Associated user is inactive"
        )
    
    return user, key_obj.organization, key_obj


async def get_user_from_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Tuple[User, Organization, 'APIKey']:
    """
    Authenticate using API key.
    Returns (user, organization, api_key_obj) tuple.
    
    Usage:
        @router.get("/endpoint")
        async def endpoint(auth = Depends(get_user_from_api_key)):
            user, organization, api_key = auth
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    return await fetch_user_and_org_from_api_key(x_api_key)


# ============= Flexible Authentication (JWT or API Key) =============

async def get_current_user_flexible(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Tuple[User, Organization, Optional['APIKey']]:
    """
    Support both JWT and API Key authentication.
    Priority: API Key > JWT
    
    Returns: (user, organization, api_key_obj_or_none)
    
    Usage:
        @router.get("/endpoint")
        async def endpoint(auth = Depends(get_current_user_flexible)):
            user, organization, api_key = auth
            # api_key is None if JWT auth was used
    """
    # Try API key first
    if x_api_key:
        user, org, api_key = await get_user_from_api_key(x_api_key)
        return user, org, api_key
    
    # Fall back to JWT
    if authorization:
        user = await get_current_user(authorization)
        organization = await get_current_organization(request, user)
        return user, organization, None
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide X-API-Key or Authorization header",
        headers={"WWW-Authenticate": "ApiKey, Bearer"}
    )


# ============= Permission Checking =============

def require_api_key_permission(required_permission: str):
    """
    Dependency factory to require specific API key permission.
    
    Usage:
        @router.post("/endpoint")
        async def endpoint(
            auth = Depends(require_api_key_permission("write"))
        ):
            user, organization, api_key = auth
    
    Args:
        required_permission: 'read', 'write', or 'admin'
    """
    async def permission_checker(
        auth: Tuple[User, Organization, Optional['APIKey']] = Depends(get_current_user_flexible)
    ) -> Tuple[User, Organization, Optional['APIKey']]:
        user, organization, api_key = auth
        
        # If JWT auth (no API key), allow all operations
        # (JWT users have their own permission system via roles)
        if api_key is None:
            return auth
        
        # Check API key permission
        if not api_key.has_permission(required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key requires '{required_permission}' permission. Current: '{api_key.permission}'"
            )
        
        return auth
    
    return permission_checker


# ============= Rate Limiting =============

@sync_to_async
def check_rate_limit(api_key: 'APIKey') -> bool:
    """
    Check if API key has exceeded rate limits.
    Returns True if within limits, raises HTTPException if exceeded.
    """
    from api_keys.models import APIKeyRateLimit
    from django.db.models import Sum
    
    now = timezone.now()
    
    # Check per-minute limit
    if api_key.rate_limit_per_minute > 0:
        minute_start = now.replace(second=0, microsecond=0)
        
        minute_limit, created = APIKeyRateLimit.objects.get_or_create(
            api_key=api_key,
            window_type='minute',
            window_start=minute_start,
            defaults={'request_count': 0}
        )
        
        if minute_limit.request_count >= api_key.rate_limit_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {api_key.rate_limit_per_minute} requests per minute",
                headers={
                    "X-RateLimit-Limit": str(api_key.rate_limit_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int((minute_start + timedelta(minutes=1)).timestamp()))
                }
            )
        
        # Increment counter
        APIKeyRateLimit.objects.filter(pk=minute_limit.pk).update(
            request_count=minute_limit.request_count + 1
        )
    
    # Check per-hour limit
    if api_key.rate_limit_per_hour > 0:
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        
        hour_limit, created = APIKeyRateLimit.objects.get_or_create(
            api_key=api_key,
            window_type='hour',
            window_start=hour_start,
            defaults={'request_count': 0}
        )
        
        if hour_limit.request_count >= api_key.rate_limit_per_hour:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {api_key.rate_limit_per_hour} requests per hour",
                headers={
                    "X-RateLimit-Limit": str(api_key.rate_limit_per_hour),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int((hour_start + timedelta(hours=1)).timestamp()))
                }
            )
        
        # Increment counter
        APIKeyRateLimit.objects.filter(pk=hour_limit.pk).update(
            request_count=hour_limit.request_count + 1
        )
    
    return True


async def check_api_key_rate_limit(
    auth: Tuple[User, Organization, Optional['APIKey']] = Depends(get_current_user_flexible)
) -> Tuple[User, Organization, Optional['APIKey']]:
    """
    Rate limiting dependency for API keys.
    JWT authentication bypasses rate limiting.
    
    Usage:
        @router.get("/endpoint")
        async def endpoint(
            auth = Depends(check_api_key_rate_limit)
        ):
            user, organization, api_key = auth
    """
    user, organization, api_key = auth
    
    # Only check rate limits for API key auth
    if api_key is not None:
        await check_rate_limit(api_key)
    
    return auth


# ============= Combined Dependency =============

def require_api_key_with_rate_limit(required_permission: str = "read"):
    """
    Combined dependency: authentication + permission + rate limiting.
    
    Usage:
        @router.post("/endpoint")
        async def endpoint(
            auth = Depends(require_api_key_with_rate_limit("write"))
        ):
            user, organization, api_key = auth
    """
    async def combined_checker(
        auth = Depends(require_api_key_permission(required_permission))
    ):
        # Rate limit check
        return await check_api_key_rate_limit(auth)
    
    return combined_checker


# ============= Usage Logging =============

@sync_to_async
def log_api_key_usage(
    api_key: 'APIKey',
    request: Request,
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: int = None
):
    """
    Log API key usage for analytics.
    Call this in middleware or after request completion.
    """
    from api_keys.models import APIKeyUsageLog
    
    # Get client IP
    ip_address = request.client.host if request.client else None
    
    # Get user agent
    user_agent = request.headers.get('user-agent')
    
    APIKeyUsageLog.objects.create(
        api_key=api_key,
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        response_time_ms=response_time_ms,
        ip_address=ip_address,
        user_agent=user_agent
    )


# ============= Example Usage in Routes =============

"""
Example 1: Simple API key auth
@router.get("/projects")
async def list_projects(
    auth = Depends(get_user_from_api_key)
):
    user, organization, api_key = auth
    # ... logic

Example 2: Require write permission
@router.post("/projects")
async def create_project(
    auth = Depends(require_api_key_permission("write"))
):
    user, organization, api_key = auth
    # ... logic

Example 3: With rate limiting
@router.get("/projects")
async def list_projects(
    auth = Depends(check_api_key_rate_limit)
):
    user, organization, api_key = auth
    # ... logic

Example 4: Combined (recommended for production)
@router.post("/projects")
async def create_project(
    auth = Depends(require_api_key_with_rate_limit("write"))
):
    user, organization, api_key = auth
    # ... logic

Example 5: Flexible (supports both JWT and API key)
@router.get("/projects")
async def list_projects(
    auth = Depends(get_current_user_flexible)
):
    user, organization, api_key = auth
    # api_key is None if JWT was used
    if api_key:
        logger.info(f"API key access: {api_key.name}")
    # ... logic
"""