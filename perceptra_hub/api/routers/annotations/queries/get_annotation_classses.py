
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
    Annotation,
    AnnotationGroup,
    AnnotationClass,
)


def xyxy2xywh(xyxy):
    return (xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])

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
    "/annotations/classes", methods=["GET"], tags=["Annotations"])
def get_annotation_classes(project_id: str):
    try:
        annotation_groups = AnnotationGroup.objects.filter(project__name=project_id)
        if not annotation_groups.exists():
            return {"classes": []}
        
        annotation_classes = AnnotationClass.objects.filter(annotation_group__in=annotation_groups).values(
            "id", "name", "color", "description"
        )
        return {"classes": list(annotation_classes)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
