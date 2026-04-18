from fastapi import APIRouter, Depends, HTTPException, status
from asgiref.sync import sync_to_async
from pydantic import BaseModel
from uuid import UUID
import logging
from api.dependencies import ProjectContext, get_project_context
from memberships.models import ProjectMembership, OrganizationMembership, Role
from users.models import CustomUser as User

router = APIRouter()
logger = logging.getLogger(__name__)


# ─── Models ───────────────────────────────────────────────────────────────────

class ProjectMemberOut(BaseModel):
    id: str
    username: str
    email: str
    first_name: str
    last_name: str
    role: str
    joined_at: str


class AddMemberRequest(BaseModel):
    user_id: str
    role: str = "viewer"


class UpdateMemberRequest(BaseModel):
    role: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

@sync_to_async
def _list_members(project) -> list:
    memberships = (
        ProjectMembership.objects
        .filter(project=project)
        .select_related('user', 'role')
        .order_by('joined_at')
    )
    return [
        ProjectMemberOut(
            id=str(m.user.id),
            username=m.user.username,
            email=m.user.email,
            first_name=m.user.first_name,
            last_name=m.user.last_name,
            role=m.role.name,
            joined_at=m.joined_at.isoformat(),
        )
        for m in memberships
    ]


@sync_to_async
def _add_member(user_id: int, role_name: str, project, organization, requesting_user) -> ProjectMemberOut:
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if not OrganizationMembership.objects.filter(user=user, organization=organization).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not a member of this organization",
        )

    if ProjectMembership.objects.filter(user=user, project=project).exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User is already a member of this project",
        )

    try:
        role = Role.objects.get(name__iexact=role_name)
    except Role.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {role_name}",
        )

    membership = ProjectMembership.objects.create(user=user, project=project, role=role)

    logger.info(
        f"User {requesting_user.username} added {user.username} "
        f"to project {project.name} as {role.name}"
    )

    return ProjectMemberOut(
        id=str(user.id),
        username=user.username,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        role=role.name,
        joined_at=membership.joined_at.isoformat(),
    )


@sync_to_async
def _update_member(user_id: int, role_name: str, project, requesting_user) -> ProjectMemberOut:
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    try:
        membership = (
            ProjectMembership.objects
            .select_related('user', 'role')
            .get(user=user, project=project)
        )
    except ProjectMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not a member of this project",
        )

    try:
        new_role = Role.objects.get(name__iexact=role_name)
    except Role.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {role_name}",
        )

    old_role = membership.role.name
    membership.role = new_role
    membership.save(update_fields=['role'])

    logger.info(
        f"User {requesting_user.username} changed {user.username}'s project role "
        f"from {old_role} to {new_role.name} in {project.name}"
    )

    return ProjectMemberOut(
        id=str(user.id),
        username=user.username,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        role=new_role.name,
        joined_at=membership.joined_at.isoformat(),
    )


@sync_to_async
def _remove_member(user_id: int, project, requesting_user) -> None:
    if requesting_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove yourself from the project",
        )

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    try:
        membership = ProjectMembership.objects.get(user=user, project=project)
    except ProjectMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not a member of this project",
        )

    membership.delete()
    logger.info(
        f"User {requesting_user.username} removed {user.username} from project {project.name}"
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/members", response_model=list[ProjectMemberOut])
async def get_project_members(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    project_ctx.require_project_access()
    members = await _list_members(project_ctx.project)
    logger.info(
        f"User {project_ctx.user.username} listed members of project {project_ctx.project.name}"
    )
    return members


@router.post("/projects/{project_id}/members", response_model=ProjectMemberOut, status_code=201)
async def add_project_member(
    project_id: UUID,
    data: AddMemberRequest,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    project_ctx.require_manage_permission()
    return await _add_member(
        int(data.user_id), data.role,
        project_ctx.project, project_ctx.organization, project_ctx.user,
    )


@router.patch("/projects/{project_id}/members/{user_id}", response_model=ProjectMemberOut)
async def update_project_member(
    project_id: UUID,
    user_id: int,
    data: UpdateMemberRequest,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    project_ctx.require_manage_permission()
    return await _update_member(user_id, data.role, project_ctx.project, project_ctx.user)


@router.delete("/projects/{project_id}/members/{user_id}", status_code=204)
async def remove_project_member(
    project_id: UUID,
    user_id: int,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    project_ctx.require_manage_permission()
    await _remove_member(user_id, project_ctx.project, project_ctx.user)
