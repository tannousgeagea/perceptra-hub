from fastapi import APIRouter, Depends, HTTPException, status
from asgiref.sync import sync_to_async
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import logging
from api.dependencies import RequestContext, get_request_context
from memberships.models import OrganizationMembership
from projects.models import Project

router = APIRouter()
logger = logging.getLogger(__name__)

class OrganizationSettingsResponse(BaseModel):
    id: str
    org_id: str
    name: str
    slug: str
    description: Optional[str] = None
    website: Optional[str] = None
    logo_url: Optional[str] = None
    settings: dict = Field(default_factory=dict)
    created_at: str
    updated_at: str


class UpdateOrganizationRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    website: Optional[str] = None
    logo_url: Optional[str] = None
    settings: Optional[dict] = None


@sync_to_async
def get_organization_settings(organization):
    """Get organization settings (admin only)."""
    return OrganizationSettingsResponse(
        id=str(organization.id),
        org_id=str(organization.org_id),
        name=organization.name,
        slug=organization.slug,
        description=organization.description if hasattr(organization, 'description') else None,
        website=organization.website if hasattr(organization, 'website') else None,
        logo_url=organization.logo_url if hasattr(organization, 'logo_url') else None,
        settings=organization.settings if hasattr(organization, 'settings') else {},
        created_at=organization.created_at.isoformat(),
        updated_at=organization.updated_at.isoformat()
    )


@sync_to_async
def update_organization_settings(organization, update_data: UpdateOrganizationRequest):
    """Update organization settings (admin only)."""
    updated_fields = []
    
    if update_data.name is not None:
        organization.name = update_data.name
        updated_fields.append('name')
    
    if update_data.description is not None:
        organization.description = update_data.description
        updated_fields.append('description')
    
    if update_data.website is not None:
        organization.website = update_data.website
        updated_fields.append('website')
    
    if update_data.logo_url is not None:
        organization.logo_url = update_data.logo_url
        updated_fields.append('logo_url')
    
    if update_data.settings is not None:
        organization.settings = update_data.settings
        updated_fields.append('settings')
    
    if updated_fields:
        organization.save(update_fields=updated_fields)
    
    return organization


@router.get("/organizations/settings", response_model=OrganizationSettingsResponse)
async def get_settings(
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Get organization settings.
    
    Requires admin or owner role.
    """
    ctx.require_role('admin', 'owner')
    
    settings = await get_organization_settings(ctx.organization)
    
    return settings


@router.patch("/organizations/settings", response_model=OrganizationSettingsResponse)
async def update_settings(
    update_data: UpdateOrganizationRequest,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Update organization settings.
    
    Requires admin or owner role.
    """
    ctx.require_role('admin', 'owner')
    
    organization = await update_organization_settings(ctx.organization, update_data)
    settings = await get_organization_settings(organization)
    
    logger.info(
        f"User {ctx.user.username} updated settings for "
        f"organization {ctx.organization.name}"
    )
    
    return settings