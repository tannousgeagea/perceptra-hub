from fastapi import APIRouter, Depends, Query
from asgiref.sync import sync_to_async
from pydantic import BaseModel
from typing import Optional, List
import logging
from api.dependencies import RequestContext, get_request_context
from memberships.models import OrganizationMembership

router = APIRouter()
logger = logging.getLogger(__name__)


class OrgMemberOut(BaseModel):
    id: str
    username: str
    email: str
    first_name: str
    last_name: str
    role: str
    status: str
    joined_at: str


class MembersResponse(BaseModel):
    total: int
    page: int
    page_size: int
    members: List[OrgMemberOut]


@sync_to_async
def _fetch_members(
    organization,
    skip: int,
    limit: int,
    search: Optional[str],
    role: Optional[str],
    status: Optional[str],
) -> tuple:
    qs = OrganizationMembership.objects.filter(
        organization=organization
    ).select_related('user', 'role')

    if search:
        from django.db.models import Q
        qs = qs.filter(
            Q(user__username__icontains=search) |
            Q(user__email__icontains=search) |
            Q(user__first_name__icontains=search) |
            Q(user__last_name__icontains=search)
        )

    if role:
        qs = qs.filter(role__name__iexact=role)

    if status:
        qs = qs.filter(status=status.lower())

    total = qs.count()
    memberships = list(qs.order_by('-joined_at')[skip:skip + limit])

    members = [
        OrgMemberOut(
            id=str(m.user.id),
            username=m.user.username,
            email=m.user.email,
            first_name=m.user.first_name,
            last_name=m.user.last_name,
            role=m.role.name,
            status=m.status,
            joined_at=m.joined_at.isoformat(),
        )
        for m in memberships
    ]
    return total, members


@router.get("/organizations/members", response_model=MembersResponse)
async def get_org_members(
    ctx: RequestContext = Depends(get_request_context),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=200, description="Page size"),
    search: Optional[str] = Query(None, description="Search by name or email"),
    role: Optional[str] = Query(None, description="Filter by role name"),
    status: Optional[str] = Query(None, description="Filter by status"),
):
    ctx.require_role('owner', 'admin')
    total, members = await _fetch_members(
        ctx.organization, skip, limit, search, role, status
    )

    logger.info(
        f"User {ctx.user.username} listed {len(members)} members "
        f"for organization {ctx.organization.name}"
    )

    return MembersResponse(
        total=total,
        page=(skip // limit) + 1,
        page_size=limit,
        members=members,
    )
