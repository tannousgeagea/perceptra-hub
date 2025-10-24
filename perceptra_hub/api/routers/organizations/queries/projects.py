from fastapi import APIRouter, Depends
from users.models import CustomUser as User
from organizations.models import Organization
from memberships.models import OrganizationMembership
from projects.models import Project
from pydantic import BaseModel
from typing import Optional
from fastapi import Path
from api.routers.auth.queries.dependencies import (
    organization_access_dependency
)

router = APIRouter()

class OrgProjectOut(BaseModel):
    id: int
    name: str
    memberCount: int
    organizationId: int

@router.get("/organizations/{org_id}/projects", response_model=list[OrgProjectOut])
def get_org_projects(
    org_id: int,
    _membership=Depends(organization_access_dependency)
):
    projects = Project.objects.filter(
        organization_id=org_id
    )

    return [
        {
            "id": p.id,
            "name": p.name,
            "memberCount": 0,
            "organizationId": org_id,
        }
        for p in projects
    ]