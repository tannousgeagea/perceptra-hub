from fastapi import APIRouter, Depends
from users.models import CustomUser as User
from organizations.models import Organization
from memberships.models import ProjectMembership
from pydantic import BaseModel
from typing import Optional
from fastapi import Path
from api.routers.auth.queries.dependencies import (
    user_project_access_dependency,
    project_admin_or_org_admin_dependency,
    project_edit_admin_or_org_admin_dependency,
)

router = APIRouter()

class ProjectMemberOut(BaseModel):
    id: str
    username: str
    email: str
    role: str

@router.get("/projects/{project_id}/members", response_model=list[ProjectMemberOut])
def get_project_members(
    project_id: str,
    _membership=Depends(project_edit_admin_or_org_admin_dependency)
):
    memberships = ProjectMembership.objects.filter(
        project__name=project_id
    ).select_related("user", "role")

    return [
        {
            "id": str(m.user.id),
            "username": m.user.username,
            "email": m.user.email,
            "role": m.role.name,
        }
        for m in memberships
    ]