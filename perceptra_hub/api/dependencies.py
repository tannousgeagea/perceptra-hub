"""
FastAPI dependencies — backward-compatible re-export hub.

All authentication logic has moved to dedicated app modules.
Import from those modules in new code; this file exists so that
all existing router imports continue to work without changes.

New code should import from:
  - users.auth              → get_current_user
  - organizations.context   → RequestContext
  - organizations.dependencies → get_request_context, require_org_role,
                                 require_org_admin, require_permission, require_scope
  - projects.context        → ProjectContext
  - projects.dependencies   → get_project_context
"""

# ─── Re-exports from domain modules ──────────────────────────────────────────

# User authentication
from users.auth import get_current_user, fetch_user_from_db          # noqa: F401

# Organization context + dependencies
from organizations.context import RequestContext                       # noqa: F401
from organizations.dependencies import (                               # noqa: F401
    get_request_context,
    require_org_role as require_organization_role,
    require_org_admin as require_organization_admin,
    require_permission,
    require_scope,
    _fetch_organization_and_verify_membership as fetch_organization_and_verify_membership,
    _fetch_organization_membership,
)

# Project context + dependencies
from projects.context import ProjectContext                            # noqa: F401
from projects.dependencies import get_project_context                  # noqa: F401

# ─── Utilities kept here (no domain home) ────────────────────────────────────

from typing import Optional, Generator
from fastapi import Depends, HTTPException, status, Header, Request
from asgiref.sync import sync_to_async

from django.contrib.auth import get_user_model
from django.conf import settings
from django.utils import timezone

from organizations.models import Organization
from memberships.models import OrganizationMembership, Role

import logging

User = get_user_model()
logger = logging.getLogger(__name__)


def get_db() -> Generator:
    """
    Database session dependency for Django ORM.
    Django manages its own connections — this is a pass-through.
    """
    try:
        yield None
    finally:
        pass


class PaginationParams:
    """Pagination parameters."""

    def __init__(self, skip: int = 0, limit: int = 100, max_limit: int = 1000):
        self.skip = max(0, skip)
        self.limit = min(max(1, limit), max_limit)
        self.page = (self.skip // self.limit) + 1 if self.limit > 0 else 1


def get_pagination(skip: int = 0, limit: int = 100) -> PaginationParams:
    """Pagination dependency."""
    return PaginationParams(skip=skip, limit=limit)


async def get_optional_user(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Optional[User]:  # type: ignore
    """
    Get current user if authenticated, None otherwise.
    Useful for endpoints that behave differently for authenticated vs anonymous users.
    """
    if not authorization:
        return None
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None


async def bypass_auth_dev(request: Request) -> tuple:
    """
    Bypass authentication for development/testing.

    WARNING: Only use this in development mode with DEBUG=True!
    REMOVE THIS DEPENDENCY IN PRODUCTION!
    """
    if not settings.DEBUG:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Development bypass only available in DEBUG mode",
        )

    logger.warning("DEVELOPMENT MODE: Authentication bypassed")

    admin_role, _ = Role.objects.get_or_create(
        name='admin',
        defaults={'description': 'Administrator'},
    )

    user, created = User.objects.get_or_create(
        username='dev-user',
        defaults={
            'email': 'dev@example.com',
            'first_name': 'Dev',
            'last_name': 'User',
            'is_active': True,
        },
    )
    if created:
        user.set_password('devpassword')
        user.save()

    organization, created = Organization.objects.get_or_create(
        slug='dev-org',
        defaults={'name': 'Development Organization'},
    )

    OrganizationMembership.objects.get_or_create(
        user=user,
        organization=organization,
        defaults={'role': admin_role},
    )

    return user, organization


# ─── Legacy functions still used by storage/queries/storage.py ───────────────

async def get_current_organization(
    request: Request,
    user: User = Depends(get_current_user),  # type: ignore
) -> Organization:
    """
    Resolve current organization from request headers/query params.

    Legacy dependency — prefer get_request_context() for new endpoints.
    Still used by storage/queries/storage.py.
    """
    org_id = request.headers.get('X-Organization-ID')
    org_slug = request.headers.get('X-Organization-Slug')

    if not org_id:
        org_id = request.query_params.get('organization_id')
    if not org_slug:
        org_slug = request.query_params.get('organization_slug')

    if not org_id and not org_slug:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization ID or slug required.",
        )

    try:
        organization = await fetch_organization_and_verify_membership(user, org_id, org_slug)
        return organization
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving organization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve organization",
        )


@sync_to_async
def _fetch_user_organization_role(user: User, organization: Organization) -> Role:  # type: ignore
    try:
        membership = OrganizationMembership.objects.get(
            user=user,
            organization=organization,
            status='active',
        )
        return membership.role
    except OrganizationMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not a member of this organization",
        )


async def get_user_organization_role(
    user: User = Depends(get_current_user),                           # type: ignore
    organization: Organization = Depends(get_current_organization),
) -> Role:
    """Legacy dependency: get user's role in the current organization."""
    return await _fetch_user_organization_role(user, organization)


def user_has_organization_permission(
    user: User,              # type: ignore
    organization: Organization,
    required_roles: list,
) -> bool:
    """Utility: check if user holds any of the required roles in the organization."""
    try:
        membership = OrganizationMembership.objects.get(
            user=user,
            organization=organization,
            status='active',
        )
        return membership.role.name in required_roles
    except OrganizationMembership.DoesNotExist:
        return False


def get_user_organizations(user: User) -> list:  # type: ignore
    """Utility: return all organizations a user belongs to."""
    memberships = OrganizationMembership.objects.filter(
        user=user,
        status='active',
    ).select_related('organization')
    return [m.organization for m in memberships]
