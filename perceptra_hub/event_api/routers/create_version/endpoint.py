
import os
import json
import time
import uuid
import importlib
import django
django.setup()
from pathlib import Path
from celery.result import AsyncResult
from django.db import transaction
from django.core.cache import cache
from starlette.responses import JSONResponse
from fastapi import Body
from fastapi import Request
from datetime import datetime
from pydantic import BaseModel
from fastapi import HTTPException, status
from fastapi.routing import APIRoute
from typing_extensions import Annotated
from fastapi import FastAPI, Depends, APIRouter, Request, Header, Response
from typing import Callable, Union, Any, Dict, AnyStr, Optional, List
from projects.models import Project, Version, VersionImage, ProjectImage

from event_api.tasks import (
    create_version
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


class ApiResponse(BaseModel):
    status: str
    task_id: str
    version_id: str

class CreateVersionRequest(BaseModel):
    project_id: str

router = APIRouter(
    prefix="/api/v1",
    tags=["Create Version"],
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)

@router.api_route(
    "/create-version", methods=["POST"], tags=["Create Version"]
)
def handle_event(
    payload: CreateVersionRequest = Body(...),
    x_request_id: Annotated[Optional[str], Header()] = None,
) -> ApiResponse:
    
    if not payload:
        raise HTTPException(status_code=400, detail="Invalid request payload")
    
    """
    Trigger version creation and zip generation in background.
    """
    # Fetch project and determine next version number (similar logic to your current API)
    project = Project.objects.filter(name=payload.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project.project_id} not found")
    
    last_version = Version.objects.filter(project=project).order_by('-version_number').first()
    next_version_number = last_version.version_number + 1 if last_version else 1
    
    images = ProjectImage.objects.filter(project=project, status="dataset", is_active=True)
    if not images.exists():
        raise HTTPException(
            status_code=400,
            detail="No reviewed images available to create a version"
        )
    
    with transaction.atomic():
        new_version = Version.objects.create(
            project=project,
            version_number=next_version_number,
            version_name=f"v{next_version_number}",
            created_at=datetime.now()
        )
        version_images = [
            VersionImage(version=new_version, project_image=img)
            for img in images
        ]
        VersionImage.objects.bulk_create(version_images)

    image_ids = list(VersionImage.objects.filter(version=new_version).values_list('id', flat=True))
    task = create_version.core.execute.apply_async(args=(new_version.id, image_ids), task_id=x_request_id)
    response_data = {
        "status": "success",
        "task_id": task.id,
        "version_id": str(new_version.id),
    }
    
    return ApiResponse(**response_data)

@router.api_route(
    "/create-version/{task_id}", methods=["GET"], tags=["Create Version"]
)
def get_task_progress(task_id: str):
    progress = cache.get(f"task_progress_{task_id}")
    if progress is None:
        # You might also query Celery's state if needed; here we simply return pending.
        return JSONResponse(status_code=status.HTTP_200_OK, content={"state": "PENDING", "progress": None})
    else:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"state": "PROGRESS", "progress": progress})
# async def get_event_status(task_id: str, x_request_id:Annotated[Optional[str], Header()] = None):
#     task_result = AsyncResult(task_id)
#     if task_result.state == 'PENDING':
#         response = {"state": task_result.state, "progress": None}
#     elif task_result.state != 'FAILURE':
#         response = {
#             "state": task_result.state,
#             "progress": task_result.info  # e.g., {'current': 50, 'total': 2000, 'status': 'In Progress'}
#         }
#     else:
#         response = {"state": task_result.state, "progress": str(task_result.info)}
#     return response
