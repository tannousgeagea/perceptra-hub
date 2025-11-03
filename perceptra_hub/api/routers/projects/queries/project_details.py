
import time
import logging
from fastapi import Request, Response
from fastapi.routing import APIRoute
from typing import Callable
from fastapi import APIRouter, HTTPException, Depends
from django.core.exceptions import ObjectDoesNotExist
from projects.models import Project, ProjectImage
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from asgiref.sync import sync_to_async
from api.dependencies import (
    ProjectContext,
    get_project_context,
)

logger = logging.getLogger(__name__)

from api.routers.projects.schemas import ProjectStatistics, UserBasicInfo, ProjectListItem

class TimedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            print(f"route duration: {duration}")
            print(f"route response: {response}")
            print(f"route response headers: {response.headers}")
            return response

        return custom_route_handler


router = APIRouter(
    prefix="/projects",
    route_class=TimedRoute,
)

@sync_to_async
def fetch_project_details(project, user):
    # Get user's role in this project
    membership = project.memberships.filter(user=user).first()
    user_role = membership.role.name if membership else 'member'
    
    # Get statistics
    image_count = project.project_images.count()
    annotation_group_count = project.annotation_groups.count()
    
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
    
    return ProjectListItem(
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
        )

@router.get("/{project_id}", response_model=ProjectListItem)
async def get_project_detail(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
    ):
    """
    Fetch project details by project_id.
    """
    return await fetch_project_details(project_ctx.project, user=project_ctx.user)
