"""
API Key authentication handler.

Provides the APIKeyAuth class for verifying API keys, enforcing rate limits,
IP restrictions, scopes, and permissions. Also includes role resolution and
the authenticate_with_api_key() dependency for FastAPI.
"""
from typing import Optional, Tuple, Annotated

from fastapi import HTTPException, status, Header, Request
from django.utils import timezone
from django.core.cache import cache
from asgiref.sync import sync_to_async

from api_keys.models import APIKey
from organizations.models import Organization

import logging

logger = logging.getLogger(__name__)


class APIKeyAuth:
    """API Key authentication handler."""

    @staticmethod
    async def verify_api_key(api_key: str) -> Tuple[APIKey, Organization]:
        """
        Verify API key and return associated API key object and organization.

        Uses Redis cache (60s TTL) keyed by key_prefix to avoid DB hits
        on repeated requests. Full key is always verified timing-safe.

        Raises:
            HTTPException 401: If key is invalid, inactive, or expired
        """
        from django.conf import settings

        expected_prefix = getattr(settings, 'API_KEY_PREFIX', 'ph')
        if not api_key.startswith(f'{expected_prefix}_'):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key format",
            )

        # Extract display prefix for cache lookup
        key_prefix = api_key[:12]

        # Check cache first (60s TTL)
        cache_key = f"api_key:{key_prefix}"
        cached_data = cache.get(cache_key)

        if cached_data:
            api_key_obj, organization = cached_data

            if not api_key_obj.verify_key(api_key):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                )

            if not api_key_obj.is_valid():
                cache.delete(cache_key)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key is inactive or expired",
                )

            return api_key_obj, organization

        # Cache miss — query database
        try:
            api_key_obj = await (
                APIKey.objects
                .select_related('organization', 'owned_by', 'created_by')
                .aget(key_prefix=key_prefix)
            )
        except APIKey.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        # Timing-safe full key verification
        if not api_key_obj.verify_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        # Check active + not expired
        if not api_key_obj.is_valid():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is inactive or expired",
            )

        organization = api_key_obj.organization

        # Cache for 60 seconds
        cache.set(cache_key, (api_key_obj, organization), 60)

        return api_key_obj, organization

    @staticmethod
    def check_rate_limit(api_key: APIKey, cache_prefix: str = "rate_limit") -> bool:
        """
        Enforce per-minute and per-hour rate limits using atomic Redis operations.

        Uses cache.add() (set-if-not-exists) + cache.incr() (atomic increment)
        to avoid read-modify-write races.

        Raises:
            HTTPException 429: If rate limit is exceeded
        """
        now = timezone.now()

        # Per-minute check
        if api_key.rate_limit_per_minute > 0:
            minute_key = f"{cache_prefix}:minute:{api_key.api_key_id}:{now.strftime('%Y%m%d%H%M')}"
            cache.add(minute_key, 0, 120)  # TTL 2 minutes for safety
            minute_count = cache.incr(minute_key)

            if minute_count > api_key.rate_limit_per_minute:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {api_key.rate_limit_per_minute} requests/min",
                    headers={"Retry-After": "60"},
                )

        # Per-hour check
        if api_key.rate_limit_per_hour > 0:
            hour_key = f"{cache_prefix}:hour:{api_key.api_key_id}:{now.strftime('%Y%m%d%H')}"
            cache.add(hour_key, 0, 7200)  # TTL 2 hours for safety
            hour_count = cache.incr(hour_key)

            if hour_count > api_key.rate_limit_per_hour:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {api_key.rate_limit_per_hour} requests/hr",
                    headers={"Retry-After": "3600"},
                )

        return True

    @staticmethod
    def check_ip_restriction(api_key: APIKey, ip_address: str) -> bool:
        """
        Check if request IP is allowed for this API key.

        Raises:
            HTTPException 403: If IP is not in the allowed list
        """
        if not api_key.is_ip_allowed(ip_address):
            logger.warning(
                f"API key {api_key.key_prefix} used from unauthorized IP: {ip_address}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key not authorized from this IP address",
            )
        return True

    @staticmethod
    def check_scope(api_key: APIKey, required_scope: str) -> bool:
        """
        Check if API key has the required scope.

        Raises:
            HTTPException 403: If key does not have the required scope
        """
        if not api_key.has_scope(required_scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key does not have required scope: {required_scope}",
            )
        return True

    @staticmethod
    def check_permission(api_key: APIKey, required_permission: str) -> bool:
        """
        Check if API key has the required permission level.

        Raises:
            HTTPException 403: If key's permission level is insufficient
        """
        if not api_key.has_permission(required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key requires '{required_permission}' permission",
            )
        return True


# ─── Role resolution ─────────────────────────────────────────────────────────

async def _resolve_api_key_role(api_key: APIKey, organization: Organization):
    """
    Derive the effective Role model instance for an API key request.

    Rules:
      1. If the key has no owned_by, map the key's declared permission to a role.
      2. If the key has an owned_by, fetch that user's OrganizationMembership
         and take the lower of (key permission rank, membership role rank).
         This prevents an admin from creating a key that exceeds the target
         user's actual privileges.
      3. If owned_by has lost their membership, drop to 'viewer' and log a
         warning — the key is still valid but maximally restricted.

    Returns a Role model instance for backward compatibility with
    ctx.has_role() / ctx.require_role().
    """
    from memberships.models import OrganizationMembership, Role

    _permission_to_rank = {'read': 1, 'write': 2, 'admin': 3}
    _rank_to_role = {1: 'viewer', 2: 'editor', 3: 'admin'}
    _role_to_rank = {'viewer': 1, 'editor': 2, 'admin': 3, 'owner': 4}
    _permission_to_role = {'read': 'viewer', 'write': 'editor', 'admin': 'admin'}

    if api_key.owned_by is None:
        # No delegation — key acts as its creator at the declared permission level
        role_name = _permission_to_role.get(api_key.permissions, 'viewer')
    else:
        # owned_by is already loaded (select_related in verify_api_key)
        # but the membership is not — fetch it async
        try:
            membership = await OrganizationMembership.objects.aget(
                user=api_key.owned_by,
                organization=organization,
                status='active',
            )
        except OrganizationMembership.DoesNotExist:
            logger.warning(
                f"API key {api_key.key_prefix}: owned_by user "
                f"{api_key.owned_by_id} has no active membership in "
                f"org {organization.slug}. Restricting to viewer."
            )
            role_name = 'viewer'
        else:
            key_rank = _permission_to_rank.get(api_key.permissions, 1)
            member_rank = _role_to_rank.get(membership.role.name, 1)
            effective = min(key_rank, member_rank)
            role_name = _rank_to_role[effective]

    # Return an actual Role model instance for compatibility
    try:
        role = await Role.objects.aget(name=role_name)
    except Role.DoesNotExist:
        # Fallback: if role doesn't exist in DB yet, create it
        role, _ = await Role.objects.aget_or_create(
            name=role_name,
            defaults={'description': f'{role_name.title()} role'},
        )

    return role


# ─── Header extraction ────────────────────────────────────────────────────────

async def get_api_key_from_header(
    x_api_key: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
) -> Optional[str]:
    """Extract API key from header."""
    return x_api_key


# ─── Full authentication dependency ──────────────────────────────────────────

async def authenticate_with_api_key(request: Request, x_api_key: str):
    """
    Full API key authentication: verify → IP check → rate limit → record usage → resolve role.

    Returns a RequestContext (imported late to avoid circular imports).
    """
    from organizations.context import RequestContext

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide X-API-Key header.",
        )

    # 1. Verify key (uses cache, timing-safe)
    api_key, organization = await APIKeyAuth.verify_api_key(x_api_key)

    # 2. Check IP restriction
    client_ip = request.client.host if request.client else "unknown"
    APIKeyAuth.check_ip_restriction(api_key, client_ip)

    # 3. Check rate limits (Redis-based, atomic)
    APIKeyAuth.check_rate_limit(api_key)

    # 4. Record usage (async, non-blocking)
    await sync_to_async(api_key.record_usage)(client_ip)

    # 5. Resolve effective role (returns Role model instance)
    role = await _resolve_api_key_role(api_key, organization)

    # 6. Determine the acting user
    user = api_key.owned_by if api_key.owned_by else api_key.created_by

    # 7. Stash on request.state for middleware access
    request.state.api_key = api_key
    request.state.auth_method = 'api_key'

    return RequestContext(
        user=user,
        organization=organization,
        role=role,
        membership=None,
        api_key=api_key,
        auth_method='api_key',
    )
