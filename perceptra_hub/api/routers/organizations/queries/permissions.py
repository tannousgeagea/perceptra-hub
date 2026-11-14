# Roles and permissions as the console sees them.
#
# memberships/roles.py is the single source of truth. These endpoints serve it
# so the console never keeps its own copy of the role lists or rules.
from typing import Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import RequestContext, get_request_context
from memberships.roles import org_permissions_for, role_catalog

router = APIRouter()


class RoleInfo(BaseModel):
    name: str
    rank: int
    description: str


class RoleCatalog(BaseModel):
    org_roles: List[RoleInfo]
    project_roles: List[RoleInfo]
    org_permissions: Dict[str, List[str]]
    project_permissions: Dict[str, List[str]]
    org_roles_with_project_bypass: List[str]


class OrgPermissions(BaseModel):
    role: str | None
    permissions: List[str]


@router.get("/roles", response_model=RoleCatalog, summary="Roles and the permission matrix")
async def get_roles(ctx: RequestContext = Depends(get_request_context)):
    """Every org and project role, in order, and which roles hold each permission."""
    return role_catalog()


@router.get(
    "/organizations/permissions",
    response_model=OrgPermissions,
    summary="What I may do in the current organization",
)
async def get_my_org_permissions(ctx: RequestContext = Depends(get_request_context)):
    role = ctx.role.name if ctx.role else None
    return {"role": role, "permissions": sorted(org_permissions_for(role))}
