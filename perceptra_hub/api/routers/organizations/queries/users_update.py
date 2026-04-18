from fastapi import APIRouter, Depends, HTTPException, status
from asgiref.sync import sync_to_async
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
import logging
from api.dependencies import RequestContext, get_request_context
from memberships.models import OrganizationMembership, Role
from users.models import CustomUser as User

router = APIRouter()
logger = logging.getLogger(__name__)


# ─── Pydantic models ──────────────────────────────────────────────────────────

class UpdateMemberRequest(BaseModel):
    role: Optional[str] = Field(None, description="New role: owner, admin, annotator, viewer")
    status: Optional[str] = Field(None, description="New status: active, inactive, pending, suspended")


class UpdateMemberResponse(BaseModel):
    message: str
    user_id: str
    username: str
    old_role: Optional[str] = None
    new_role: Optional[str] = None
    old_status: Optional[str] = None
    new_status: Optional[str] = None
    changes: list[str]


class InviteMemberRequest(BaseModel):
    email: str = Field(..., description="Email address of the user to add")
    role: str = Field("viewer", description="Role to assign: owner, admin, annotator, viewer")


class InviteMemberResponse(BaseModel):
    message: str
    user_id: str
    username: str
    email: str
    role: str


# ─── Update role / status ─────────────────────────────────────────────────────

@sync_to_async
def _update_member(
    target_user_id: int,
    new_role_name: Optional[str],
    new_status: Optional[str],
    organization,
    requesting_user: User,
) -> dict:
    if not new_role_name and not new_status:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one of 'role' or 'status' must be provided",
        )

    try:
        target_user = User.objects.get(id=target_user_id)
    except User.DoesNotExist:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    try:
        membership = OrganizationMembership.objects.select_related('role').get(
            user=target_user, organization=organization
        )
    except OrganizationMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not a member of this organization",
        )

    changes = []
    old_role = membership.role.name
    old_status = membership.status
    updates = {}

    if new_role_name:
        if target_user.id == requesting_user.id and old_role == 'owner':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change your own role as owner",
            )
        try:
            new_role = Role.objects.get(name__iexact=new_role_name)
        except Role.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {new_role_name}",
            )
        if old_role != new_role.name:
            membership.role = new_role
            updates['role'] = new_role
            changes.append(f"role: {old_role} → {new_role.name}")

    if new_status:
        valid_statuses = ['active', 'inactive', 'pending', 'suspended']
        if new_status.lower() not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {new_status}. Must be one of: {', '.join(valid_statuses)}",
            )
        if target_user.id == requesting_user.id and new_status.lower() != 'active':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change your own status to inactive/suspended",
            )
        if old_status != new_status.lower():
            membership.status = new_status.lower()
            updates['status'] = new_status.lower()
            changes.append(f"status: {old_status} → {new_status.lower()}")

    if updates:
        membership.save(update_fields=list(updates.keys()))
        logger.info(
            f"User {requesting_user.username} updated {target_user.username} "
            f"in org {organization.name}: {', '.join(changes)}"
        )
    else:
        changes.append("no changes made")

    new_role_obj = updates.get('role', membership.role)
    return {
        "user_id": str(target_user.id),
        "username": target_user.username or target_user.email,
        "old_role": old_role if new_role_name else None,
        "new_role": new_role_obj.name if new_role_name else None,
        "old_status": old_status if new_status else None,
        "new_status": updates.get('status', membership.status) if new_status else None,
        "changes": changes,
    }


@router.patch("/organizations/users/{user_id}", response_model=UpdateMemberResponse)
async def update_member(
    user_id: int,
    data: UpdateMemberRequest,
    ctx: RequestContext = Depends(get_request_context),
):
    ctx.require_role('admin', 'owner')
    result = await _update_member(user_id, data.role, data.status, ctx.organization, ctx.user)
    return UpdateMemberResponse(
        message=f"User updated: {', '.join(result['changes'])}",
        **result,
    )


# ─── Invite / add member ──────────────────────────────────────────────────────

@sync_to_async
def _add_member(email: str, role_name: str, organization, requesting_user: User) -> dict:
    try:
        user = User.objects.get(email__iexact=email.strip())
    except User.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No account found for {email}. The user must sign up first.",
        )

    if OrganizationMembership.objects.filter(user=user, organization=organization).exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User is already a member of this organization",
        )

    try:
        role = Role.objects.get(name__iexact=role_name)
    except Role.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {role_name}",
        )

    OrganizationMembership.objects.create(
        user=user,
        organization=organization,
        role=role,
        status='active',
        invited_by=requesting_user,
    )

    logger.info(
        f"User {requesting_user.username} added {user.email} "
        f"to org {organization.name} as {role.name}"
    )

    return {
        "user_id": str(user.id),
        "username": user.username or email.split('@')[0],
        "email": user.email,
        "role": role.name,
    }


@router.post("/organizations/members", response_model=InviteMemberResponse, status_code=status.HTTP_201_CREATED)
async def add_member(
    data: InviteMemberRequest,
    ctx: RequestContext = Depends(get_request_context),
):
    ctx.require_role('admin', 'owner')
    result = await _add_member(data.email, data.role, ctx.organization, ctx.user)
    return InviteMemberResponse(
        message=f"{result['email']} added to the organization as {result['role']}",
        **result,
    )


# ─── Remove member ────────────────────────────────────────────────────────────

@sync_to_async
def _remove_member(target_user_id: int, organization, requesting_user: User) -> None:
    try:
        target_user = User.objects.get(id=target_user_id)
    except User.DoesNotExist:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if target_user.id == requesting_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove yourself from the organization",
        )

    try:
        membership = OrganizationMembership.objects.select_related('role').get(
            user=target_user, organization=organization
        )
    except OrganizationMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not a member of this organization",
        )

    if membership.role.name == 'owner':
        owner_count = OrganizationMembership.objects.filter(
            organization=organization, role__name='owner'
        ).count()
        if owner_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove the last owner of the organization",
            )

    membership.delete()
    logger.info(
        f"User {requesting_user.username} removed {target_user.username or target_user.email} "
        f"from org {organization.name}"
    )


@router.delete("/organizations/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    user_id: int,
    ctx: RequestContext = Depends(get_request_context),
):
    ctx.require_role('admin', 'owner')
    await _remove_member(user_id, ctx.organization, ctx.user)
