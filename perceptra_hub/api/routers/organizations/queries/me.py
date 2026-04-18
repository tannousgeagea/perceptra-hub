from fastapi import APIRouter, Depends
from asgiref.sync import sync_to_async
from pydantic import BaseModel
from typing import Optional
import logging
from api.dependencies import RequestContext, get_request_context
from memberships.models import OrganizationMembership
from projects.models import Project

router = APIRouter()
logger = logging.getLogger(__name__)


class OrganizationOut(BaseModel):
    id: str
    org_id: str
    name: str
    slug: str
    description: Optional[str] = None
    user_count: int
    project_count: int
    current_user_role: str


@sync_to_async
def _fetch_org_summary(organization, role_name: str) -> OrganizationOut:
    user_count = OrganizationMembership.objects.filter(
        organization=organization, status='active'
    ).count()
    project_count = Project.objects.filter(
        organization=organization, is_deleted=False
    ).count()
    return OrganizationOut(
        id=str(organization.id),
        org_id=str(organization.org_id),
        name=organization.name,
        slug=organization.slug,
        description=getattr(organization, 'description', None),
        user_count=user_count,
        project_count=project_count,
        current_user_role=role_name,
    )


@router.get("/organizations/me", response_model=OrganizationOut)
async def get_my_organization(ctx: RequestContext = Depends(get_request_context)):
    ctx.require_role('owner', 'admin')
    logger.info(f"User {ctx.user.username} fetching organization summary: {ctx.organization.name}")
    return await _fetch_org_summary(ctx.organization, ctx.role_name or "member")
