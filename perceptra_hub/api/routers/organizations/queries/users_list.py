from fastapi import APIRouter, Depends, Query
from asgiref.sync import sync_to_async
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import logging
from api.dependencies import RequestContext, get_request_context
from memberships.models import OrganizationMembership
from users.models import CustomUser as User

router = APIRouter()
logger = logging.getLogger(__name__)


# Pydantic models
class OrganizationInfo(BaseModel):
    id: str
    name: str
    slug: str
    role: str


class OrganizationUserOut(BaseModel):
    id: str
    username: str
    email: str
    first_name: str
    last_name: str
    role: str
    status: str
    created_at: str
    last_active: Optional[str] = None
    avatar: Optional[str] = None
    organizations: List[OrganizationInfo] = []


class OrganizationUsersResponse(BaseModel):
    total: int
    page: int
    page_size: int
    users: List[OrganizationUserOut]


@sync_to_async
def get_organization_users_list(
    organization,
    skip: int,
    limit: int,
    role_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    search: Optional[str] = None
):
    """Get all users in an organization with filtering."""
    # Base queryset
    queryset = OrganizationMembership.objects.filter(
        organization=organization
    ).select_related('user', 'role')
    
    # Filter by role
    if role_filter:
        queryset = queryset.filter(role__name__iexact=role_filter)
    
    # Filter by user status
    if status_filter:
        queryset = queryset.filter(status=status_filter.lower())
    
    # Search by name or email
    if search:
        from django.db.models import Q
        queryset = queryset.filter(
            Q(user__username__icontains=search) |
            Q(user__email__icontains=search) |
            Q(user__first_name__icontains=search) |
            Q(user__last_name__icontains=search)
        )
    
    # Get total count
    total = queryset.count()
    
    # Get paginated results
    memberships = list(queryset.order_by('-joined_at')[skip:skip + limit])
    
    # Build response
    users = []
    for membership in memberships:
        user = membership.user
        
        # Get user's organizations
        user_orgs = OrganizationMembership.objects.filter(
            user=user
        ).select_related('organization', 'role')[:5]  # Limit to 5 for performance
        
        organizations = [
            OrganizationInfo(
                id=str(org_mem.organization.id),
                name=org_mem.organization.name,
                slug=org_mem.organization.slug,
                role=org_mem.role.name
            )
            for org_mem in user_orgs
        ]
        
        users.append(OrganizationUserOut(
            id=str(user.id),
            username=user.username,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            role=membership.role.name,
            status=membership.status,
            created_at=membership.joined_at.isoformat(),
            last_active=user.last_login.isoformat() if user.last_login else None,
            avatar=None,  # Add avatar field to User model if needed
            organizations=organizations
        ))
    
    return {
        "total": total,
        "users": users,
        "skip": skip,
        "limit": limit
    }


@router.get("/organizations/users", response_model=OrganizationUsersResponse)
async def list_organization_users(
    ctx: RequestContext = Depends(get_request_context),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    role: Optional[str] = Query(None, description="Filter by role (owner, admin, annotator, viewer)"),
    status: Optional[str] = Query(None, description="Filter by status (active, inactive)"),
    search: Optional[str] = Query(None, description="Search by username, email, or name")
):
    """
    Get all users in the current organization.
    
    Requires organization ID/slug in header or query parameter.
    
    Query Parameters:
    - skip: Pagination offset
    - limit: Number of results per page
    - role: Filter by role name
    - status: Filter by user status (active/inactive)
    - search: Search in username, email, first/last name
    """
    
    ctx.require_role('owner', 'admin')
    result = await get_organization_users_list(
        ctx.organization,
        skip,
        limit,
        role,
        status,
        search
    )
    
    logger.info(
        f"User {ctx.user.username} listed {len(result['users'])} users "
        f"from organization {ctx.organization.name}"
    )
    
    return OrganizationUsersResponse(
        total=result["total"],
        page=(result["skip"] // result["limit"]) + 1,
        page_size=result["limit"],
        users=result["users"]
    )