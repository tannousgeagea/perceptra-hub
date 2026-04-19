from fastapi import APIRouter, Depends, Query
from asgiref.sync import sync_to_async
from pydantic import BaseModel
from typing import Optional, List
import logging
from api.dependencies import RequestContext, get_request_context
from projects.models import Project

router = APIRouter()
logger = logging.getLogger(__name__)


class OrgProjectOut(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    member_count: int
    organization_id: str
    is_active: bool
    created_at: str


class ProjectsResponse(BaseModel):
    total: int
    projects: List[OrgProjectOut]


@sync_to_async
def _fetch_projects(organization, search: Optional[str]) -> tuple:
    qs = Project.objects.filter(
        organization=organization,
        is_deleted=False,
    )

    if search:
        qs = qs.filter(name__icontains=search)

    total = qs.count()
    projects = list(qs.prefetch_related('memberships').order_by('-created_at'))

    result = []
    for p in projects:
        try:
            member_count = p.memberships.count()
        except Exception:
            member_count = 0

        result.append(OrgProjectOut(
            id=str(p.project_id),
            name=p.name,
            description=getattr(p, 'description', None),
            member_count=member_count,
            organization_id=str(organization.id),
            is_active=p.is_active,
            created_at=p.created_at.isoformat(),
        ))

    return total, result


@router.get("/organizations/projects", response_model=ProjectsResponse)
async def get_org_projects(
    ctx: RequestContext = Depends(get_request_context),
    search: Optional[str] = Query(None, description="Filter projects by name"),
):
    ctx.require_role('owner', 'admin')
    total, projects = await _fetch_projects(ctx.organization, search)

    logger.info(
        f"User {ctx.user.username} listed {total} projects "
        f"for organization {ctx.organization.name}"
    )

    return ProjectsResponse(total=total, projects=projects)
