import os
import time
import django
django.setup()
from fastapi import APIRouter
from fastapi import FastAPI, HTTPException, status
from fastapi import Request, Response
from pydantic import BaseModel
from fastapi.routing import APIRoute
from typing import Callable, Optional
from typing import List, Optional, Dict
from datetime import datetime
from django.db.models import Count
from fastapi import status as http_status

from projects.models import (
    Project,
    ProjectMetadata,
    Version,
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
    
class CurrentVersion(BaseModel):
    id: int
    version_number: str
    description: Optional[str]
    created_at: datetime
    image_count: int


class ProjectSummary(BaseModel):
    id: int
    name: str
    description: Optional[str]
    type: str
    visibility_level: str
    created_at: datetime
    updated_at: datetime
    current_version: Optional[CurrentVersion]
    metadata: Dict[str, str]

router = APIRouter(
    route_class=TimedRoute,
)
@router.api_route(
    "/analytics/summary", methods=["GET"], tags=["Analytics"])
def get_project_summary(
    project_id:str
):
    try:
        try:
            project = Project.objects.select_related('project_type', 'visibility')\
                                    .get(name=project_id, is_active=True)
        except Project.DoesNotExist:
            raise HTTPException(status_code=404, detail="Project not found")

        latest_version = (
            Version.objects
            .filter(project=project)
            .order_by("-version_number")
            .annotate(image_count=Count("version_images"))
            .first()
        )

        version_data = None
        if latest_version:
            version_data = CurrentVersion(
                id=latest_version.id,
                version_number=f"v{latest_version.version_number}",
                description=latest_version.description if latest_version.description else f"Version v{latest_version.version_number}",
                created_at=latest_version.created_at,
                image_count=latest_version.image_count
            )

        metadata_qs = ProjectMetadata.objects.filter(project=project)
        metadata_dict = {meta.key: meta.value for meta in metadata_qs}

        return ProjectSummary(
            id=project.id,
            name=project.name,
            description=project.description,
            type=project.project_type.name if project.project_type else "Unknown",
            visibility_level=project.visibility.name if project.visibility else "Unknown",
            created_at=project.created_at,
            updated_at=project.last_edited,
            current_version=version_data,
            metadata=metadata_dict
        )
    
    except HTTPException as e:
        raise  e
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )