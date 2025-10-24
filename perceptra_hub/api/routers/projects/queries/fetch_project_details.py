
import time
from fastapi import Request, Response
from fastapi.routing import APIRoute
from typing import Callable
from fastapi import APIRouter, HTTPException
from django.core.exceptions import ObjectDoesNotExist
from projects.models import Project, ProjectImage
from pydantic import BaseModel
from typing import Optional

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
    route_class=TimedRoute,
)
class ProjectDetailResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    thumbnail_url: Optional[str]
    project_type: Optional[str]
    last_edited: str
    visibility: str
    created_at: str
    is_active: bool

@router.get("/projects/{project_id}", response_model=ProjectDetailResponse)
def get_project_detail(project_id: str):
    """
    Fetch project details by project_id.
    """
    try:
        project = Project.objects.select_related("project_type", "visibility").get(name=project_id)
    except ObjectDoesNotExist:
        raise HTTPException(status_code=404, detail="Project not found.")

    return ProjectDetailResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        thumbnail_url=ProjectImage.objects.filter(project=project).first().image.image_file.url if ProjectImage.objects.filter(project=project).exists() else None,
        project_type=project.project_type.name if project.project_type else None,
        last_edited=project.last_edited.isoformat(),
        visibility=project.visibility.name if project.visibility else "Unknown",
        created_at=project.created_at.isoformat(),
        is_active=project.is_active,
    )
