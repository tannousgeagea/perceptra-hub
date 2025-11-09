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
    
class ProjectStats(BaseModel):
    total_images: int
    annotated_images: int
    reviewed_images: int
    finalized_images: int

router = APIRouter(
    route_class=TimedRoute,
)
@router.api_route(
    "/analytics/stats", methods=["GET"], tags=["Analytics"])
def get_project_stats(
    project_id:str
):
    try:
        try:
            project = Project.objects.get(name=project_id, is_active=True)
        except Project.DoesNotExist:
            raise HTTPException(status_code=404, detail="Project not found")

        qs = ProjectImage.objects.filter(project=project, is_active=True)

        return ProjectStats(
            total_images=qs.count(),
            annotated_images=qs.filter(annotated=True).count(),
            reviewed_images=qs.filter(reviewed=True).count(),
            finalized_images=qs.filter(finalized=True).count()
        )
    
    except HTTPException as e:
        raise  e
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )