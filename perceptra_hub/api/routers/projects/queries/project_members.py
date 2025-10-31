from fastapi import APIRouter, Depends
from asgiref.sync import sync_to_async
from pydantic import BaseModel
from uuid import UUID
import logging
from api.dependencies import ProjectContext, get_project_context
from memberships.models import ProjectMembership

router = APIRouter()
logger = logging.getLogger(__name__)

class ProjectMemberOut(BaseModel):
    id: int
    username: str
    email: str
    first_name: str
    last_name: str
    role: str
    joined_at: str

@sync_to_async
def get_project_members_list(project):
    """Get all members of a project."""
    memberships = ProjectMembership.objects.filter(
        project=project
    ).select_related('user', 'role').order_by('joined_at')
    
    return [
        ProjectMemberOut(
            id=m.user.id,
            username=m.user.username,
            email=m.user.email,
            first_name=m.user.first_name,
            last_name=m.user.last_name,
            role=m.role.name,
            joined_at=m.joined_at.isoformat()
        )
        for m in memberships
    ]

@router.get("/projects/{project_id}/members", response_model=list[ProjectMemberOut])
async def get_project_members(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """
    Get all members of a project.
    
    Access automatically checked:
    - Organization admins/owners: ✅
    - Project members: ✅
    - Others: ❌
    """
    # Access is already checked by get_project_context dependency
    # No need for manual permission checks!
    
    members = await get_project_members_list(project_ctx.project)
    
    logger.info(
        f"User {project_ctx.user.username} accessed members of "
        f"project {project_ctx.project.name}"
    )
    
    return members