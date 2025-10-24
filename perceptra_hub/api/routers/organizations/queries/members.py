from fastapi import APIRouter, Depends
from users.models import CustomUser as User
from organizations.models import Organization
from memberships.models import OrganizationMembership
from pydantic import BaseModel
from typing import Optional
from fastapi import Path
from api.routers.auth.queries.dependencies import (
    organization_access_dependency
)

router = APIRouter()

class OrgMemberOut(BaseModel):
    id: int
    username: str
    email: str
    role: str

@router.get("/organizations/{org_id}/members", response_model=list[OrgMemberOut])
def get_org_members(
    org_id: int,
    _membership=Depends(organization_access_dependency)
):
    memberships = OrganizationMembership.objects.filter(
        organization_id=org_id
    ).select_related("user", "role")

    return [
        {
            "id": m.user.id,
            "username": m.user.username,
            "email": m.user.email,
            "role": m.role.name,
        }
        for m in memberships
    ]