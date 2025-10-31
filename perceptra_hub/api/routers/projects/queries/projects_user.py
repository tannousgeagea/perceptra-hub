import time
from fastapi import APIRouter, Depends, HTTPException
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

# Pydantic models
class UserBasicInfo(BaseModel):
    id: int
    username: str
    email: str
    first_name: str
    last_name: str

class ProjectStatistics(BaseModel):
    total_images: int
    total_annotations: int = 0
    annotation_groups: int

class ProjectListItem(BaseModel):
    id: int
    project_id: str
    name: str
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    project_type_name: str
    visibility_name: str
    is_active: bool
    statistics: ProjectStatistics
    created_by: Optional[UserBasicInfo] = None
    updated_by: Optional[UserBasicInfo] = None
    created_at: str
    last_edited: str
    user_role: str = Field(..., description="Current user's role in this project")

@sync_to_async
def get_user_projects(user: User, organization):
    """Get all projects for user in organization."""
    projects = Project.objects.filter(
        organization=organization,
        is_active=True,
        is_deleted=False,
        memberships__user=user
    ).select_related(
        'project_type',
        'visibility',
        'created_by',
        'updated_by',
        'organization'
    ).prefetch_related(
        'project_images',
        'annotation_groups',
        'memberships__role'
    ).distinct()
    
    result = []
    for project in projects:
        # Get user's role in this project
        membership = project.memberships.filter(user=user).first()
        user_role = membership.role.name if membership else 'member'
        
        # Get statistics
        image_count = project.project_images.count()
        annotation_group_count = project.annotation_groups.count()
        
        # Get or generate thumbnail
        thumbnail = project.thumbnail_url
        if not thumbnail:
            first_image = project.project_images.first()
            if first_image and first_image.image:
                try:
                    thumbnail = first_image.image.get_download_url(expiration=3600)
                except Exception as e:
                    logger.warning(f"Failed to get thumbnail for project {project.id}: {e}")
                    thumbnail = None
        
        # Build creator info
        created_by = None
        if project.created_by:
            created_by = UserBasicInfo(
                id=project.created_by.id,
                username=project.created_by.username,
                email=project.created_by.email,
                first_name=project.created_by.first_name,
                last_name=project.created_by.last_name
            )
        
        # Build updater info
        updated_by = None
        if project.updated_by:
            updated_by = UserBasicInfo(
                id=project.updated_by.id,
                username=project.updated_by.username,
                email=project.updated_by.email,
                first_name=project.updated_by.first_name,
                last_name=project.updated_by.last_name
            )
        
        result.append(ProjectListItem(
            id=project.id,
            project_id=str(project.project_id),
            name=project.name,
            description=project.description,
            thumbnail_url=thumbnail,
            project_type_name=project.project_type.name,
            visibility_name=project.visibility.name,
            is_active=project.is_active,
            statistics=ProjectStatistics(
                total_images=image_count,
                total_annotations=0,  # Add when annotations are implemented
                annotation_groups=annotation_group_count
            ),
            created_by=created_by,
            updated_by=updated_by,
            created_at=project.created_at.isoformat(),
            last_edited=project.last_edited.isoformat(),
            user_role=user_role
        ))
    
    return result

@router.get("/projects/me", response_model=List[ProjectListItem])
async def get_my_projects(
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Get all projects for the current user in the current organization.
    
    Requires organization ID/slug in header or query parameter.
    Returns projects where the user is a member.
    """
    projects = await get_user_projects(ctx.user, ctx.organization)
    return projects