import time
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from django.db.models import Count
from users.models import CustomUser as User
from projects.models import Project, ProjectImage
from fastapi.routing import APIRoute
from fastapi import Request, Response
from typing import Callable, Optional
from api.routers.auth.queries.dependencies import (
    user_project_access_dependency,
    project_admin_or_org_admin_dependency,
    get_current_user
)

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
    route_class=TimedRoute
)

class AssignedUserOut(BaseModel):
    id: int
    username: str
    email: str
    avatar: Optional[str] = None

class ProjectOut(BaseModel):
    id: int
    name: str
    lastEdited: str
    images:int
    thumbnail: Optional[str] = None
    description: Optional[str] = None
    assignedUser: Optional[AssignedUserOut] = None

@router.get("/projects/me/", response_model=List[ProjectOut])
def get_projects_for_user(
    user=Depends(get_current_user),
):
    projects = Project.objects.filter(
        is_active=True, 
        memberships__user=user
    ).prefetch_related('project_images').distinct()
    

    result = []
    for project in projects:
        # Get image count and thumbnail more efficiently
        project_images = list(project.project_images.all())
        image_count = len(project_images)
        thumbnail = project.thumbnail_url
        if not thumbnail:
            thumbnail = project_images[0].image.image_file.url if project_images else None
            project.save(update_fields=['thumbnail_url'])

        
        result.append(ProjectOut(
            id=project.pk,
            name=project.name,
            description=project.description,
            images=image_count,
            assignedUser=None,  # Project model doesn't have assignee field
            lastEdited=project.created_at.isoformat(),
            thumbnail=thumbnail,
        ))
    
    return result