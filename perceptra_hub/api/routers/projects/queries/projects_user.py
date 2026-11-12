import time
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel, Field
from django.db.models import Count
from asgiref.sync import sync_to_async
from users.models import CustomUser as User
from projects.models import Project, ProjectImage
from fastapi.routing import APIRoute
from fastapi import Request, Response
from typing import Callable
from api.dependencies import get_request_context, RequestContext
from api.routers.projects.schemas import ProjectListItem
from api.routers.projects.serializers import serialize_project, with_list_data
from memberships.roles import OrgRole
import logging

logger = logging.getLogger(__name__)

class TimedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            return response
        return custom_route_handler

router = APIRouter(route_class=TimedRoute)

@sync_to_async
def get_user_projects(user: User, organization, include_inactive: bool = False, org_role: Optional[str] = None):
    """
    Projects the user can open in the organization: all of them for org
    admins/owners (who get access to every project, see
    projects.resolution.fetch_project_with_access_check), otherwise the ones
    they're a member of.
    """
    sees_all = org_role in (OrgRole.ADMIN, OrgRole.OWNER)
    projects = Project.objects.filter(organization=organization, is_deleted=False)
    if not sees_all:
        projects = projects.filter(memberships__user=user)
    if not include_inactive:
        projects = projects.filter(is_active=True)
    # Membership join can duplicate rows; dedupe by id before annotating.
    projects = Project.objects.filter(pk__in=projects.values("pk")).order_by("-created_at")
    fallback_role = org_role if sees_all else "member"
    return [serialize_project(p, user, fallback_role) for p in with_list_data(projects, user)]


@router.get("/projects/user/me", response_model=List[ProjectListItem])
async def get_my_projects(
    ctx: RequestContext = Depends(get_request_context),
    include_inactive: bool = Query(
        False, description="Also return projects switched to inactive (the Projects page shows both)."
    ),
):
    """
    Get all projects for the current user in the current organization.
    
    Requires organization ID/slug in header or query parameter.
    Returns every project for org admins/owners, otherwise the user's projects.
    """
    org_role = ctx.role.name if ctx.role else None
    return await get_user_projects(ctx.user, ctx.organization, include_inactive, org_role)
