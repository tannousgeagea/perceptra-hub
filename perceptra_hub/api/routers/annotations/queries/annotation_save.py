
import os
import math
import uuid
import time
import django
import shutil
django.setup()
from datetime import datetime, timedelta
from datetime import date, timezone
from typing import Callable, Optional, Dict, AnyStr, Any, List
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import Depends, Form, Body
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi import status
from pathlib import Path
from django.db import transaction
from pydantic import BaseModel

from projects.models import (
    Project,
    ProjectImage,
)

from annotations.models import (
    Annotation,
    AnnotationClass,
    AnnotationGroup,
    AnnotationType
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

class AnnotationData(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float
    label: str
    color: str

class SaveAnnotationsRequest(BaseModel):
    annotations:AnnotationData

@router.api_route(
    "/annotations/{project_id}/{image_id}", methods=["POST"], tags=["Annotations"])
def save_annotations(project_id: str, image_id: str, request: AnnotationData):
    try:
        project_image = ProjectImage.objects.get(project__name=project_id, image__image_id=image_id)

        with transaction.atomic():
            annotation_data = request
            annotation_class = AnnotationClass.objects.filter(
                annotation_group__project=project_image.project, name=annotation_data.label
            ).first()

            if not annotation_class:
                annotation_group = AnnotationGroup.objects.filter(project=project_image.project).first()
                if not annotation_group:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No annotation group found for project {project_id}",
                    )
                    
                max_class_id = AnnotationClass.objects.filter(
                    annotation_group=annotation_group
                    ).order_by('-class_id').first().class_id or 0
                
                print(max_class_id)
                annotation_class = AnnotationClass.objects.create(
                    annotation_group=annotation_group,
                    name=annotation_data.label,
                    class_id=max_class_id + 1,
                    color=annotation_data.color,
                )
            
            annotation_type = AnnotationType.objects.get(name="bounding_boxes")
            if not Annotation.objects.filter(annotation_uid=annotation_data.id).exists():
                Annotation.objects.create(
                    project_image=project_image,
                    annotation_type=annotation_type,
                    annotation_class=annotation_class,
                    data=[
                        annotation_data.x,
                        annotation_data.y,
                        annotation_data.x + annotation_data.width,
                        annotation_data.y + annotation_data.height,   
                    ],
                    annotation_uid=f"{annotation_data.id}",
                    annotation_source="manual",
                )
                
                if not project_image.annotated or project_image.status == "unannotated":
                    project_image.annotated = True
                    project_image.status = "annotated"
                    project_image.save(update_fields=["annotated", "status"])
                    
                return {"message": "Annotations created successfully."}
            
            annotation = Annotation.objects.filter(annotation_uid=annotation_data.id).first()
            annotation.data = [
                    annotation_data.x,
                    annotation_data.y,
                    annotation_data.x + annotation_data.width,
                    annotation_data.y + annotation_data.height,   
            ]
            
            annotation.annotation_class = annotation_class
            annotation.annotation_source = "manual"
            annotation.save()

            if not project_image.annotated or project_image.status == "unannotated":
                project_image.annotated = True
                project_image.status = "annotated"
                project_image.save(update_fields=["annotated", "status"])

            return {"message": "Annotations updated successfully.", "status": project_image.annotated}
        
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))