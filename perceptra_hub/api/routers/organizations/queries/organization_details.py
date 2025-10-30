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


# Pydantic models
class OrganizationStatistics(BaseModel):
    total_members: int
    active_members: int
    inactive_members: int
    pending_members: int
    total_projects: int
    active_projects: int
    total_images: int


class OrganizationMemberSummary(BaseModel):
    id: str
    username: str
    email: str
    first_name: str
    last_name: str
    role: str
    status: str
    joined_at: str


class OrganizationDetailsResponse(BaseModel):
    id: str
    org_id: str
    name: str
    slug: str
    description: Optional[str] = None
    website: Optional[str] = None
    logo_url: Optional[str] = None
    created_at: str
    updated_at: str
    
    # Current user's info
    current_user_role: str
    current_user_status: str
    current_user_joined_at: str
    
    # Statistics
    statistics: OrganizationStatistics
    
    # Recent members (last 5)
    recent_members: List[OrganizationMemberSummary]


@sync_to_async
def get_organization_details(user, organization):
    """Get detailed information about the organization."""
    
    # Get current user's membership
    try:
        current_membership = OrganizationMembership.objects.select_related('role').get(
            user=user,
            organization=organization
        )
    except OrganizationMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not a member of this organization"
        )
    
    # Get statistics
    all_memberships = OrganizationMembership.objects.filter(
        organization=organization
    )
    
    total_members = all_memberships.count()
    active_members = all_memberships.filter(status='active').count()
    inactive_members = all_memberships.filter(status='inactive').count()
    pending_members = all_memberships.filter(status='pending').count()
    
    all_projects = Project.objects.filter(
        organization=organization,
        is_deleted=False
    )
    total_projects = all_projects.count()
    active_projects = all_projects.filter(is_active=True).count()
    
    # Count images (if Image model has organization FK)
    from images.models import Image
    total_images = Image.objects.filter(organization=organization).count()
    
    # Get recent members (last 5)
    recent_memberships = OrganizationMembership.objects.filter(
        organization=organization
    ).select_related('user', 'role').order_by('-joined_at')[:5]
    
    recent_members = [
        OrganizationMemberSummary(
            id=str(m.user.id),
            username=m.user.username,
            email=m.user.email,
            first_name=m.user.first_name,
            last_name=m.user.last_name,
            role=m.role.name,
            status=m.status,
            joined_at=m.joined_at.isoformat()
        )
        for m in recent_memberships
    ]
    
    return OrganizationDetailsResponse(
        id=str(organization.id),
        org_id=str(organization.org_id),
        name=organization.name,
        slug=organization.slug,
        description=organization.description if hasattr(organization, 'description') else None,
        website=organization.website if hasattr(organization, 'website') else None,
        logo_url=organization.logo_url if hasattr(organization, 'logo_url') else None,
        created_at=organization.created_at.isoformat(),
        updated_at=organization.updated_at.isoformat(),
        
        # Current user info
        current_user_role=current_membership.role.name,
        current_user_status=current_membership.status,
        current_user_joined_at=current_membership.joined_at.isoformat(),
        
        # Statistics
        statistics=OrganizationStatistics(
            total_members=total_members,
            active_members=active_members,
            inactive_members=inactive_members,
            pending_members=pending_members,
            total_projects=total_projects,
            active_projects=active_projects,
            total_images=total_images
        ),
        
        # Recent members
        recent_members=recent_members
    )


@router.get("/organizations/details", response_model=OrganizationDetailsResponse)
async def get_organization_info(
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Get detailed information about the current organization.
    
    Requires organization ID/slug in header or query parameter.
    
    Returns:
    - Organization basic information
    - Current user's role and status in the organization
    - Statistics (members, projects, images)
    - Recent members (last 5 joined)
    """
    details = await get_organization_details(ctx.user, ctx.organization)
    
    logger.info(
        f"User {ctx.user.username} retrieved details for "
        f"organization {ctx.organization.name}"
    )
    
    return details