import os
import math
import uuid
import time
import django
import shutil
django.setup()
from django.db.models import Q
from datetime import datetime, timedelta
from datetime import time as dtime
from datetime import date, timezone
from typing import Callable, Optional, Dict, AnyStr, Any
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi import status
from pathlib import Path
from pydantic import BaseModel

from projects.models import (
    Project,
    ProjectImage,
    Version,
    VersionImage
)

from annotations.models import (
    Annotation
)


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

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
    responses={404: {"description": "Not found"}},
)

@router.api_route(
    "/projects/{project_id}/versions", methods=["GET"], tags=["Versions"]
)
def get_project_versions(project_id: str):
    """
    Fetch all versions for a specific project.
    """
    try:
        project = Project.objects.filter(name=project_id).first()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found."
            )

        versions = Version.objects.filter(project=project).order_by("-created_at")
        if not versions.exists():
            return []


        response = []
        for version in versions:
            version_images = VersionImage.objects.filter(version=version)
            response.append(
                {
                    "id": version.id,
                    "version_number": version.version_number,
                    "name": f"v{version.version_number}",
                    "created_at": version.created_at.isoformat(),
                    "description": version.description,
                    "count_images": len(version_images),
                    "count_train": len(version_images.filter(project_image__mode__mode="train")),
                    "count_val": len(version_images.filter(project_image__mode__mode='valid')),
                }
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching versions: {str(e)}"
        )