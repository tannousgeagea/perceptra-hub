import os
import zipfile
import time
import django
import shutil
django.setup()
from django.db.models import Q
from datetime import datetime, timedelta
from typing import Callable, Dict, AnyStr, Any
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi import status
from pathlib import Path
from pydantic import BaseModel
from django.core.files.base import ContentFile
from storages.backends.azure_storage import AzureStorage
from starlette.responses import FileResponse
from common_utils.data.annotation.core import save_annotations_into_txtfile
from projects.models import (
    Version, 
    ProjectImage, 
    Project,
    VersionImage,
)

from annotations.models import Annotation

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
    "/projects/{project_id}/versions/{version_id}/download", methods=["GET"], tags=["Versions"]
    )
def download_version(project_id: str, version_id: int):
    try:
        # Fetch version and associated images
        project = Project.objects.filter(name=project_id).first()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found."
            )

        version = Version.objects.filter(project=project, version_number=version_id).first()
        if not version or not version.version_file:
            raise HTTPException(status_code=404, detail="Version not found")
        
        return {"download_url": version.version_file.url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
