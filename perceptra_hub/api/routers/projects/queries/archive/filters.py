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

from projects.models import (
    Project,
    ProjectImage,
)

from annotations.models import (
    AnnotationGroup
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
)

@router.api_route(
    "/projects/{project_id}/filters", methods=["GET"], tags=["Projects"]
)
def get_filters(project_id:str=None):
    project = Project.objects.filter(name=project_id).first()
    items = [
        { "key": "1", "value": "low" },
        { "key": "2", "value": "meduim"},
        { "key": "3", "value": "high" },
        ]
    
    if project:
        annotation_group = AnnotationGroup.objects.filter(project=project).first()
        annotation_classes = annotation_group.classes.all()
        items = [
            {
                "key": str(annotation.class_id),
                "value": annotation.name,
            } for annotation in annotation_classes
        ]

    return {
        "filters": [
            {
                "key": "filename",
                "title": "Filename",
                "description": "",
                "placeholder": "Filter by Filename",
                "type": "text",
                "items": None,
            },
            {
                "key": "annotation_count",
                "title": "Annotation Count",
                "description": "",
                "placeholder": "annotation count",
                "type": "text",
                "items": None,
            },
            {
                "key": "mode",
                "title": "Split",
                "description": "",
                "placeholder": "Split",
                "type": "select",
                "items": [
                    { "key": "train", "value": "Train" },
                    { "key": "valid", "value": "Valid" }
                ]
            },
            {
                "key": "classes",
                "title": "Classes",
                "description": "",
                "placeholder": "Classes",
                "type": "select",
                "items": items
            },
        ]
    }

