import os
import time
import django
django.setup()
from fastapi import APIRouter
from fastapi import FastAPI, HTTPException, status
from fastapi import Request, Response
from pydantic import BaseModel
from fastapi.routing import APIRoute
from typing import Callable, Optional
from typing import List, Optional, Dict
from datetime import date
from datetime import datetime
from django.db.models.functions import TruncMonth
from django.db.models import Count
from fastapi import status as http_status

from projects.models import (
    Project,
    ProjectImage,
    ProjectMetadata,
    Version,
)

from annotations.models import (
    Annotation,
    AnnotationClass,
    AnnotationGroup,
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
    
class AnnotationClassItem(BaseModel):
    id: int
    name: str
    color: str
    count: int


class AnnotationGroupItem(BaseModel):
    id: int
    name: str
    classes: List[AnnotationClassItem]


AnnotationGroupList = List[AnnotationGroupItem]

router = APIRouter(
    route_class=TimedRoute,
)
@router.api_route(
    "/analytics/annotationgroups", methods=["GET"], tags=["Analytics"])
def get_annotation_groups(
    project_id:str
):
    try:
        try:
            project = Project.objects.get(name=project_id, is_active=True)
        except Project.DoesNotExist:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project_images = ProjectImage.objects.filter(project=project, is_active=True)

        annotation_groups = (
            AnnotationGroup.objects
            .filter(project=project)
            .prefetch_related("classes")
        )

        annotations = Annotation.objects.filter(project_image__in=project_images, is_active=True)

        # Count annotations per class
        class_counts = annotations.values("annotation_class").annotate(count=Count("id"))
        class_count_map = {entry["annotation_class"]: entry["count"] for entry in class_counts}

        results = []
        for group in annotation_groups:
            class_items = []
            for cls in group.classes.all():
                class_items.append(AnnotationClassItem(
                    id=cls.id,
                    name=cls.name,
                    color=cls.color or "#888",
                    count=class_count_map.get(cls.id, 0)
                ))
            results.append(AnnotationGroupItem(
                id=group.id,
                name=group.name,
                classes=class_items
            ))
        
        return results
    
    except HTTPException as e:
        raise  e
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )