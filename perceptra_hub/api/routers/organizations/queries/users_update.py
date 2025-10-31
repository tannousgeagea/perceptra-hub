from fastapi import APIRouter, Depends, HTTPException, status
from asgiref.sync import sync_to_async
from pydantic import BaseModel, Field
from typing import Optional
import logging
from api.dependencies import RequestContext, get_request_context
from memberships.models import OrganizationMembership, Role
from users.models import CustomUser as User

router = APIRouter()
logger = logging.getLogger(__name__)


# Pydantic models
class UpdateUserRequest(BaseModel):
    role: Optional[str] = Field(None, description="New role: owner, admin, annotator, viewer")
    status: Optional[str] = Field(None, description="New status: active, inactive, pending, suspended")


class UserUpdateResponse(BaseModel):
    message: str
    user_id: str
    username: str
    old_role: Optional[str] = None
    new_role: Optional[str] = None
    old_status: Optional[str] = None
    new_status: Optional[str] = None
    changes: list[str]


# ============= Combined Update Endpoint =============

@sync_to_async
def update_user_in_organization(
    target_user_id: int,
    new_role_name: Optional[str],
    new_status: Optional[str],
    organization,
    requesting_user: User
):
    """Update a user's role and/or status in the organization."""
    
    # Validate at least one field is provided
    if not new_role_name and not new_status:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one of 'role' or 'status' must be provided"
        )
    
    # Get target user
    try:
        target_user = User.objects.get(id=target_user_id)
    except User.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {target_user_id} not found"
        )
    
    # Get target user's membership
    try:
        membership = OrganizationMembership.objects.select_related('role').get(
            user=target_user,
            organization=organization
        )
    except OrganizationMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User is not a member of this organization"
        )
    
    changes = []
    old_role = membership.role.name
    old_status = membership.status
    updates = {}
    
    # Update role if provided
    if new_role_name:
        # Prevent self-demotion from owner
        if target_user.id == requesting_user.id and old_role == 'owner':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change your own role as owner"
            )
        
        # Get new role
        try:
            new_role = Role.objects.get(name__iexact=new_role_name)
        except Role.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {new_role_name}. Must be one of: owner, admin, annotator, viewer"
            )
        
        if old_role != new_role.name:
            membership.role = new_role
            updates['role'] = new_role
            changes.append(f"role changed from {old_role} to {new_role.name}")
    
    # Update status if provided
    if new_status:
        # Validate status
        valid_statuses = ['active', 'inactive', 'pending', 'suspended']
        if new_status.lower() not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {new_status}. Must be one of: {', '.join(valid_statuses)}"
            )
        
        # Prevent self-suspension/deactivation
        if target_user.id == requesting_user.id and new_status.lower() != 'active':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change your own status to inactive/suspended"
            )
        
        if old_status != new_status.lower():
            membership.status = new_status.lower()
            updates['status'] = new_status.lower()
            changes.append(f"status changed from {old_status} to {new_status.lower()}")
    
    # Save changes
    if updates:
        membership.save(update_fields=list(updates.keys()))
        
        logger.info(
            f"User {requesting_user.username} updated {target_user.username} "
            f"in organization {organization.name}: {', '.join(changes)}"
        )
    else:
        changes.append("no changes made")
    
    return {
        "user_id": str(target_user.id),
        "username": target_user.username,
        "old_role": old_role if new_role_name else None,
        "new_role": updates.get('role', membership.role).name if new_role_name else None,
        "old_status": old_status if new_status else None,
        "new_status": updates.get('status', membership.status) if new_status else None,
        "changes": changes
    }


@router.patch("/organizations/users/{user_id}", response_model=UserUpdateResponse)
async def update_user(
    user_id: int,
    data: UpdateUserRequest,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Update a user's role and/or status in the organization.
    
    Requires admin or owner role.
    
    Restrictions:
    - Cannot change your own role if you're an owner
    - Cannot change your own status to inactive/suspended
    
    Request body can include:
    - role: New role (owner, admin, annotator, viewer)
    - status: New status (active, inactive, pending, suspended)
    - Both fields are optional, but at least one must be provided
    """
    # Require admin or owner
    ctx.require_role('admin', 'owner')
    
    result = await update_user_in_organization(
        user_id,
        data.role,
        data.status,
        ctx.organization,
        ctx.user
    )
    
    return UserUpdateResponse(
        message=f"User updated successfully: {', '.join(result['changes'])}",
        user_id=result['user_id'],
        username=result['username'],
        old_role=result['old_role'],
        new_role=result['new_role'],
        old_status=result['old_status'],
        new_status=result['new_status'],
        changes=result['changes']
    )