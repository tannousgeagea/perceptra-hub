# organizations/dependencies.py
#
# FastAPI dependency functions for organization-level authentication.
# Composes: users/auth.py + organizations/resolution.py + organizations/context.py
# + api_keys/auth.py (no circular import — api_keys now imports from organizations.context)

from typing import Optional, Annotated

from fastapi import Depends, HTTPException, status, Header, Request
from asgiref.sync import sync_to_async

from django.contrib.auth import get_user_model

from memberships.models import OrganizationMembership, Role
from memberships.roles import OrgRole, roles_for_permission
from organizations.models import Organization
from organizations.context import RequestContext
from users.auth import get_current_user
from api_keys.auth import authenticate_with_api_key

import logging

User = get_user_model()
logger = logging.getLogger(__name__)


# ─── Internal helpers ─────────────────────────────────────────────────────────

@sync_to_async
def _fetch_organization_and_verify_membership(
    user: User,          # type: ignore
    org_id: Optional[str],
    org_slug: Optional[str],
) -> Organization:
    """
    Fetch organization and verify the user is an active member.
    Raises 404 if not found, 403 if not a member.
    """
    try:
        if org_id:
            organization = Organization.objects.get(org_id=org_id, is_active=True)
        else:
            organization = Organization.objects.get(slug=org_slug, is_active=True)
    except Organization.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    try:
        OrganizationMembership.objects.get(
            user=user,
            organization=organization,
            status='active',
        )
    except OrganizationMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User is not an active member of organization '{organization.name}'",
        )

    return organization


@sync_to_async
def _fetch_organization_membership(
    user: User,          # type: ignore
    organization: Organization,
) -> OrganizationMembership:
    """Fetch the active membership record (with role) for user+org."""
    try:
        return OrganizationMembership.objects.select_related('role').get(
            user=user,
            organization=organization,
            status='active',
        )
    except OrganizationMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User is not an active member of organization '{organization.name}'",
        )


# ─── Main context dependency ──────────────────────────────────────────────────

async def get_request_context(
    request: Request,
    authorization: Annotated[Optional[str], Header()] = None,
    x_api_key: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
    x_organization_id: Annotated[Optional[str], Header(alias="X-Organization-ID")] = None,
    x_organization_slug: Annotated[Optional[str], Header(alias="X-Organization-Slug")] = None,
) -> RequestContext:
    """
    Resolve the full request context (user + organization + role).

    Authentication priority: API Key > JWT Bearer token.

    - API Key: organization is implicit from the key; no extra header needed.
    - JWT:     organization resolved from X-Organization-ID / X-Organization-Slug
               header (or organization_id / organization_slug query params).
    """
    # ── API Key authentication (priority) ──
    if x_api_key:
        return await authenticate_with_api_key(request, x_api_key)

    # ── JWT authentication ──
    if authorization:
        user = await get_current_user(authorization)

        org_id = x_organization_id or request.query_params.get('organization_id')
        org_slug = x_organization_slug or request.query_params.get('organization_slug')

        if not org_id and not org_slug:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Organization ID or slug required. "
                    "Provide via X-Organization-ID header, X-Organization-Slug header, "
                    "or organization_id / organization_slug query parameter."
                ),
            )

        organization = await _fetch_organization_and_verify_membership(user, org_id, org_slug)
        membership = await _fetch_organization_membership(user, organization)

        return RequestContext(
            user=user,
            organization=organization,
            role=membership.role,
            membership=membership,
            auth_method='jwt',
        )

    # ── No credentials ──
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide X-API-Key or Authorization header.",
        headers={"WWW-Authenticate": "ApiKey, Bearer"},
    )


# ─── Role / permission dependency factories ───────────────────────────────────

def require_org_role(*required_roles: str):
    """
    Dependency factory: require the user to hold one of the specified org roles.

    Usage:
        @router.delete("/org")
        async def delete_org(ctx = Depends(require_org_role(OrgRole.OWNER))):
            ...
    """
    async def _check(ctx: RequestContext = Depends(get_request_context)) -> RequestContext:
        if not ctx.has_role(*required_roles):
            current = ctx.role_name or "none"
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(required_roles)}. Your role: {current}",
            )
        return ctx

    return _check


def require_org_admin():
    """
    Dependency factory: require org admin or owner role.

    Usage:
        @router.post("/members")
        async def invite_member(ctx = Depends(require_org_admin())):
            ...
    """
    return require_org_role(OrgRole.ADMIN, OrgRole.OWNER)


def require_permission(permission: str):
    """
    Dependency factory: require a named permission level.

    Works with both JWT (role-based) and API Key (permission-level) auth.
    The permission is mapped to allowed roles via PERMISSION_TO_ORG_ROLES.

    Usage:
        @router.post("/projects")
        async def create_project(ctx = Depends(require_permission("edit"))):
            ...
    """
    allowed_roles = roles_for_permission(permission, scope="org")

    async def _check(ctx: RequestContext = Depends(get_request_context)) -> RequestContext:
        if ctx.api_key:
            from api_keys.auth import APIKeyAuth
            APIKeyAuth.check_permission(ctx.api_key, permission)
        else:
            if not ctx.has_role(*allowed_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires '{permission}' permission",
                )
        return ctx

    return _check


def require_scope(scope: str):
    """
    Dependency factory: require a specific API key scope.

    Only enforced for API key auth; JWT auth bypasses scope checks.

    Usage:
        @router.get("/projects")
        async def list_projects(ctx = Depends(require_scope("projects:read"))):
            ...
    """
    async def _check(ctx: RequestContext = Depends(get_request_context)) -> RequestContext:
        if ctx.api_key:
            from api_keys.auth import APIKeyAuth
            APIKeyAuth.check_scope(ctx.api_key, scope)
        return ctx

    return _check
