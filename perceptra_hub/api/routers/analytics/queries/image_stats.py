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
    
class ImageUploadTrend(BaseModel):
    date: date
    count: int

class ImageStats(BaseModel):
    total: int
    status_breakdown: Dict[str, int]
    upload_trend: List[ImageUploadTrend]

router = APIRouter(
    route_class=TimedRoute,
)
@router.api_route(
    "/analytics/imagestats", methods=["GET"], tags=["Analytics"])
def get_image_stats(
    project_id:str
):
    try:
        try:
            project = Project.objects.get(name=project_id, is_active=True)
        except Project.DoesNotExist:
            raise HTTPException(status_code=404, detail="Project not found")

        qs = ProjectImage.objects.filter(project=project, is_active=True)

        # Status breakdown
        status_counts = qs.values("status").annotate(count=Count("id"))
        status_breakdown = {
            "unannotated": 0,
            "annotated": 0,
            "reviewed": 0,
            "dataset": 0,
            "null_marked": qs.filter(marked_as_null=True).count()
        }
        for entry in status_counts:
            status_breakdown[entry["status"]] = entry["count"]

        # Upload trend by month
        upload_counts = (
            qs.annotate(month=TruncMonth("added_at"))
            .values("month")
            .annotate(count=Count("id"))
            .order_by("month")
        )

        upload_trend = [
            ImageUploadTrend(date=entry["month"].date(), count=entry["count"])
            for entry in upload_counts
        ]

        return ImageStats(
            total=qs.count(),
            status_breakdown=status_breakdown,
            upload_trend=upload_trend
        )
    
    except HTTPException as e:
        raise  e
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )