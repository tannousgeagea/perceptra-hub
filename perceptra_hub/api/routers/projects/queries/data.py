import os
import math
import uuid
import time
import django
import shutil
django.setup()
from datetime import datetime, timedelta
from datetime import date, timezone
from typing import Callable, Optional, Dict, AnyStr, Any
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import Depends, Form, Body
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi import status
from pathlib import Path

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


from projects.models import (
    Project,
    ProjectImage,
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
    route_class=TimedRoute,
)

@router.api_route(
    "/projects", methods=["GET"], tags=["Projects"]
)
def get_projects(
    response: Response,
    ):
    results = {}
    try:
        
        projects = Project.objects.filter(is_active=True)
        results = {
            "data": [
                {
                    "id": project.id,
                    "name": project.name,
                    "lastEdited": project.last_edited,
                    "images": len(ProjectImage.objects.filter(project=project)),
                    "thumbnail": ProjectImage.objects.filter(project=project).first().image.image_file.url if ProjectImage.objects.filter(project=project).exists() else None,
                } for project in projects
            ]
        }
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    

