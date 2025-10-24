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
from datetime import date
from datetime import datetime
from django.db.models.functions import TruncMonth
from django.db.models import Count
from fastapi import status as http_status

from projects.models import (
    Project,
    ProjectImage,
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
    

class VersionItem(BaseModel):
    id: int
    version_number: str
    description: str
    created_at: datetime
    image_count: int
    download_url: str


VersionList = List[VersionItem]

router = APIRouter(
    route_class=TimedRoute,
)
@router.api_route(
    "/analytics/versions", methods=["GET"], tags=["Analytics"])
def get_project_versions(
    project_id:str
):
    try:
        try:
            project = Project.objects.get(name=project_id, is_active=True)
        except Project.DoesNotExist:
            raise HTTPException(status_code=404, detail="Project not found")

        versions = (
            Version.objects
            .filter(project=project)
            .annotate(image_count=Count("version_images"))
            .order_by("-version_number")
        )

        version_items = []
        for version in versions:
            version_items.append(
                VersionItem(
                    id=version.id,
                    version_number=f"v{version.version_number}",
                    description=version.description or f"Version v{version.version_number} created at {version.created_at.strftime('%Y-%m-%d')}",
                    created_at=version.created_at,
                    image_count=version.image_count,
                    download_url=version.version_file.url if version.version_file else "#"
                )
            )
    
        return version_items
    except HTTPException as e:
        raise  e
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )