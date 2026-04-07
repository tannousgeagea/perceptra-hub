# organizations/resolution.py
#
# Pure organization lookup — no membership verification, no user context.
# Membership checking belongs in organizations/dependencies.py.

from typing import Optional
from typing import Annotated

from fastapi import Header, Query, HTTPException
from organizations.models import Organization


async def resolve_organization(
    x_organization_id: Annotated[Optional[str], Header(alias="X-Organization-ID")] = None,
    x_organization_slug: Annotated[Optional[str], Header(alias="X-Organization-Slug")] = None,
    organization_id: Annotated[Optional[str], Query()] = None,
    organization_slug: Annotated[Optional[str], Query()] = None,
) -> Organization:
    """
    Resolve an Organization from headers or query parameters.

    Priority: Header > Query parameter
    Priority: ID > Slug
    """
    if x_organization_id:
        return await _get_organization_by_id(x_organization_id)

    if organization_id:
        return await _get_organization_by_id(organization_id)

    if x_organization_slug:
        return await _get_organization_by_slug(x_organization_slug)

    if organization_slug:
        return await _get_organization_by_slug(organization_slug)

    raise HTTPException(
        status_code=400,
        detail={
            "error": "Organization identification required",
            "options": [
                "Provide X-Organization-ID header",
                "Provide X-Organization-Slug header",
                "Provide organization_id query parameter",
                "Provide organization_slug query parameter",
            ],
        },
    )


async def _get_organization_by_id(organization_id: str) -> Organization:
    """Fetch an active Organization by its public UUID."""
    try:
        org = await Organization.objects.aget(org_id=organization_id, is_active=True)
        return org
    except Organization.DoesNotExist:
        raise HTTPException(
            status_code=404,
            detail=f"Organization with ID '{organization_id}' not found or inactive",
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Organization ID format: '{organization_id}'",
        )


async def _get_organization_by_slug(slug: str) -> Organization:
    """Fetch an active Organization by its slug."""
    try:
        org = await Organization.objects.aget(slug=slug, is_active=True)
        return org
    except Organization.DoesNotExist:
        raise HTTPException(
            status_code=404,
            detail=f"Organization with slug '{slug}' not found or inactive",
        )
