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


from images.models import Image
from tenants.models import (
    Tenant,
    Plant,
    EdgeBox,
)
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
    "/projects/{project_id}/dataset-info", methods=["GET"], tags=["Projects"]
)
def get_dataset_info(project_id: str):
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

        images = ProjectImage.objects.filter(project=project, reviewed=True, is_active=True)
        if not images:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No Images Found for {project_id}."
            )

        response = {
            "total_images": images.count(),
            "count_train": images.filter(mode__mode='train').count(),
            "count_val": images.filter(mode__mode='valid').count(),
            "data": [
                {
                    "image_url": img.image.image_file.url,
                    "image_name": img.image.image_name,
                } for img in images[:10]
            ]
        }

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching versions: {str(e)}"
        )
