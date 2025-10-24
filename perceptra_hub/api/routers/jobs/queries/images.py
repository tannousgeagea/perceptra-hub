import os
import math
import json
import django
import time
from typing import Optional
from fastapi import APIRouter, Depends, Path, Query, HTTPException, Response
from pydantic import BaseModel
from django.db.models import Count, Q
from fastapi.routing import APIRoute

from projects.models import ProjectImage
from jobs.models import Job, JobImage
from annotations.models import Annotation
from api.routers.auth.queries.dependencies import get_current_user

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

class TimedRoute(APIRoute):
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request):
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            return response

        return custom_route_handler

router = APIRouter(route_class=TimedRoute)

class JobImageOut(BaseModel):
    project_id: str
    image_id: str
    image_name: str
    image_url: str
    created_at: str
    plant: Optional[str]
    edge_box: Optional[str]
    location: Optional[str]
    sub_location: Optional[str]
    annotated: bool
    reviewed: bool
    annotations: list

@router.get("/jobs/{job_id}/images")
def get_job_images(
    job_id: int = Path(...),
    status: Optional[str] = Query(None),
    user_filters: Optional[str] = Query(None),
    items_per_page: int = 50,
    page: int = 1,
):
    try:
        job = Job.objects.select_related("project").filter(id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        filters_dict = {}
        if user_filters:
            try:
                filters_dict = json.loads(user_filters)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format for user_filters")

        lookup_filters = Q()
        if status:
            lookup_filters &= Q(project_image__status=status)
        lookup_filters &= Q(project_image__is_active=True)

        def filter_mapping(key, value):
            if not value or value == "all":
                return None
            if key == "mode":
                return Q(project_image__mode__mode=value)
            if key == "filename":
                return Q(project_image__image__image_name=value)
            if key == "classes":
                return Q(project_image__annotations__annotation_class__class_id=value, project_image__annotations__is_active=True)
            if key == "annotation_count":
                return Q(annotation_count=value)
            return None

        for key, value in filters_dict.items():
            q_filter = filter_mapping(key, value)
            if q_filter:
                lookup_filters &= q_filter

        queryset = JobImage.objects.filter(job=job).filter(lookup_filters).annotate(
            annotation_count=Count("project_image__annotations", filter=Q(project_image__annotations__is_active=True))
        ).order_by("-project_image__added_at").distinct()

        total_images = queryset.count()

        data = []
        paginated = queryset[(page - 1) * items_per_page: page * items_per_page]
        for ji in paginated:
            pi = ji.project_image
            annotations = Annotation.objects.filter(project_image=pi, is_active=True)
            data.append({
                'project_id': pi.project.name,
                'image_id': pi.image.image_id,
                'image_name': pi.image.image_name,
                'image_url': 'http://localhost:81' + pi.image.image_file.url if os.getenv('DJANGO_STORAGE') != 'azure' else pi.image.image_file.url,
                'created_at': pi.image.created_at.strftime(DATETIME_FORMAT),
                'plant': pi.image.sensorbox.edge_box.plant.plant_name if pi.image.sensorbox else None,
                'edge_box': pi.image.sensorbox.sensor_box_name if pi.image.sensorbox else None,
                'location': pi.image.sensorbox.edge_box.edge_box_location if pi.image.sensorbox else None,
                'sub_location': pi.image.sensorbox.sensor_box_location if pi.image.sensorbox else None,
                'annotated': pi.annotated,
                'reviewed': pi.reviewed,
                'annotations': [
                    {
                        "class_id": ann.annotation_class.class_id,
                        "class_name": ann.annotation_class.name,
                        "xyxyn": ann.data,
                    } for ann in annotations
                ]
            })

        job_image_ids = JobImage.objects.filter(job=job).values_list("project_image_id", flat=True)
        return {
            "total_record": total_images,
            "pages": math.ceil(total_images / items_per_page),
            "unannotated": ProjectImage.objects.filter(id__in=job_image_ids, status="unannotated", is_active=True).count(),
            "annotated": ProjectImage.objects.filter(id__in=job_image_ids, status="annotated", is_active=True).count(),
            "reviewed": ProjectImage.objects.filter(id__in=job_image_ids, status="reviewed", is_active=True).count(),
            "user_filters": str(filters_dict),
            "data": data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
