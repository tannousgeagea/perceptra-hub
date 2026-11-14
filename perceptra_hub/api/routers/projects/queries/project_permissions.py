# What the signed-in person may do in one project, computed by the same rule
# the API enforces (ProjectContext.permissions → memberships.roles).
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import ProjectContext, get_project_context

router = APIRouter()


class ProjectPermissions(BaseModel):
    org_role: str | None
    project_role: str | None
    permissions: List[str]


@router.get(
    "/projects/{project_id}/permissions",
    response_model=ProjectPermissions,
    summary="What I may do in this project",
)
async def get_my_project_permissions(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    return {
        "org_role": project_ctx.org_role.name if project_ctx.org_role else None,
        "project_role": project_ctx.project_role.name if project_ctx.project_role else None,
        "permissions": sorted(project_ctx.permissions()),
    }
