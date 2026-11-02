
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

from api.routers.projects.schemas import ProjectListItem
from api.routers.projects.serializers import serialize_project

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
    return serialize_project(project, user)

@router.get("/{project_id}", response_model=ProjectListItem)
async def get_project_detail(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
    ):
    """
    Fetch project details by project_id.
    """
    return await fetch_project_details(project_ctx.project, user=project_ctx.user)
