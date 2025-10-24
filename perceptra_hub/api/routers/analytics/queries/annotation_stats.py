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
    
class AnnotationClassDistribution(BaseModel):
    id: int
    name: str
    color: str
    count: int

class AnnotationStats(BaseModel):
    total: int
    class_distribution: List[AnnotationClassDistribution]
    source_breakdown: Dict[str, int]
    review_status: Dict[str, int]
    average_per_image: float

router = APIRouter(
    route_class=TimedRoute,
)
@router.api_route(
    "/analytics/annotationstats", methods=["GET"], tags=["Analytics"])
def get_annotation_stats(
    project_id:str
):
    try:
        try:
            project = Project.objects.get(name=project_id, is_active=True)
        except Project.DoesNotExist:
            raise HTTPException(status_code=404, detail="Project not found")

        project_images = ProjectImage.objects.filter(project=project, is_active=True)
        annotations = Annotation.objects.filter(project_image__in=project_images, is_active=True)

        total = annotations.count()

        # Class distribution
        class_counts = (
            annotations.values("annotation_class")
            .annotate(count=Count("id"))
        )

        # Get class info
        class_ids = [c["annotation_class"] for c in class_counts]
        class_objs = AnnotationClass.objects.filter(id__in=class_ids).select_related("annotation_group")
        class_map = {c.id: c for c in class_objs}

        class_distribution = [
            AnnotationClassDistribution(
                id=cid["annotation_class"],
                name=class_map[cid["annotation_class"]].name,
                color=class_map[cid["annotation_class"]].color or "#888",
                count=cid["count"]
            )
            for cid in class_counts if cid["annotation_class"] in class_map
        ]

        # Source breakdown
        source_counts = annotations.values("annotation_source").annotate(count=Count("id"))
        source_breakdown = {s["annotation_source"]: s["count"] for s in source_counts}
        if "manual" not in source_breakdown:
            source_breakdown["manual"] = 0
        if "prediction" not in source_breakdown:
            source_breakdown["prediction"] = 0
        source_breakdown = {
            "manual": source_breakdown["manual"],
            "model_generated": source_breakdown["prediction"]
        }

        # Review status (you can adjust the definitions if you have explicit statuses)
        review_status = {
            "pending": annotations.filter(reviewed=False).count(),
            "approved": annotations.filter(reviewed=True, feedback_provided=False).count(),
            "rejected": annotations.filter(feedback_provided=True).count()
        }

        image_count = project_images.count()
        average = total / image_count if image_count > 0 else 0.0

        return AnnotationStats(
            total=total,
            class_distribution=class_distribution,
            source_breakdown=source_breakdown,
            review_status=review_status,
            average_per_image=round(average, 2)
        )
    
    except HTTPException as e:
        raise  e
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )