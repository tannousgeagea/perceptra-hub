
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
    Annotation
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
    "/annotations/{project_id}/{image_id}", methods=["GET"], tags=["Annotations"])
def get_annotations(
    response:Response,
    project_id:str,
    image_id: str
    ):
    try:
        project = Project.objects.filter(name=project_id).first()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found."
            )
        project_image = ProjectImage.objects.filter(project=project, image__image_id=image_id).first()
        if not project_image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image with ID {image_id} not found."
            )
            
        annotations = Annotation.objects.filter(project_image=project_image, is_active=True)

        return [
            {
                "id": annotation.id,
                "project_image_id": annotation.project_image.id,
                "annotation_type": annotation.annotation_type.name,
                "annotation_class": annotation.rating.name if annotation.rating else annotation.annotation_class.name,
                "data": {
                    "id": annotation.annotation_uid if annotation.annotation_uid else str(annotation.id),
                    "x": annotation.data[0],
                    "y":annotation.data[1],
                    "width": annotation.data[2] - annotation.data[0],
                    "height": annotation.data[3] - annotation.data[1],
                    "label": annotation.annotation_class.name,
                    "color": annotation.annotation_class.color,
                    },
                "created_at": annotation.created_at.isoformat(),
                "created_by": annotation.created_by,
                "reviewed": annotation.reviewed,
                "is_active": annotation.is_active,
            }
            for annotation in annotations
        ]

    except ProjectImage.DoesNotExist:
        raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))