import os
import math
import uuid
import time
import django
import shutil
django.setup()
from django.db.models import Q, Count
from datetime import datetime, timedelta
from datetime import time as dtime
from datetime import date, timezone
from typing import Callable, Optional, Dict, AnyStr, Any, List
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query
from fastapi.routing import APIRoute
from fastapi import status
from pathlib import Path
from pydantic import BaseModel

from images.models import Image
from tenants.models import (
    Tenant,
    Plant,
    EdgeBox,
)

from django.db.models import Prefetch
from annotations.models import Annotation

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
    responses={404: {"description": "Not found"}},
)


def parse_query_list(query_list: List[str]) -> dict:
    if not query_list:
        return {}
    
    parsed = {}
    for item in query_list:
        if ":" in item:
            key, value = item.split(":", 1)
            parsed[key.strip()] = value.strip()
    return parsed


@router.api_route(
    "/images", methods=["GET"], tags=["Images"]
)
def list_tagged_images(
    limit: int = Query(20, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    name: str = Query(None),
    source: str = Query(None),
    tag: str = Query(None),
    query: Optional[List[str]] = Query(None),
):
    queryset = (
        Image.objects.prefetch_related(
            "image_tags__tag",
            "projects"
        ).select_related("sensorbox__edge_box").order_by('-created_at')
    )

    queryset = queryset.prefetch_related(
        "image_tags__tag",
        Prefetch("projects__annotations", queryset=Annotation.objects.filter(is_active=True).select_related("annotation_class")),
        "projects__project"
    ).select_related("sensorbox__edge_box")

    if name:
        queryset = queryset.filter(image_name__icontains=name)

    if source:
        queryset = queryset.filter(
            sensorbox__edge_box__edge_box_location__icontains=source
        ) | queryset.filter(sensorbox__sensor_box_location__icontains=source
        ) | queryset.filter(source_of_origin__icontains=source)

    if tag:
        queryset = queryset.filter(image_tags__tag__name__icontains=tag)

    parsed_query = parse_query_list(query)
    if "tenant" in parsed_query:
        queryset = queryset.filter(
            sensorbox__edge_box__plant__tenant__name__icontains=parsed_query['tenant']
        )
    
    if "location" in parsed_query:
        queryset = queryset.filter(
            sensorbox__sensor_box_location__icontains=parsed_query['location']
        )

    if "tag" in parsed_query:
        queryset = queryset.filter(image_tags__tag__name__icontains=parsed_query["tag"])

    if "project" in parsed_query:
        queryset = queryset.filter(projects__project__name__icontains=parsed_query["project"])

    if "annotation_class" in parsed_query:
        queryset = queryset.filter(projects__annotations__annotation_class__name__icontains=parsed_query["annotation_class"])

    if "exclude_dataset" in parsed_query and parsed_query["exclude_dataset"].lower() == "true":
        queryset = queryset.annotate(
            dataset_status_count=Count(
                'projects',
                filter=Q(projects__status="dataset")
            )
        ).filter(dataset_status_count=0)

    try:
        if "created_at" in parsed_query:
            queryset = queryset.filter(created_at=datetime.strptime(parsed_query["created_at"], "%Y-%m-%d"))
        if "created_at__gte" in parsed_query:
            queryset = queryset.filter(created_at__gte=datetime.strptime(parsed_query["created_at__gte"], "%Y-%m-%d"))
        if "created_at__lte" in parsed_query:
            queryset = queryset.filter(created_at__lte=datetime.strptime(parsed_query["created_at__lte"], "%Y-%m-%d"))
        if "created_at__range" in parsed_query:
            range_vals = parsed_query["created_at__range"].split(",")
            if len(range_vals) == 2:
                start = datetime.strptime(range_vals[0], "%Y-%m-%d")
                end = datetime.strptime(range_vals[1], "%Y-%m-%d")
                queryset = queryset.filter(created_at__range=(start, end))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format in query. Use YYYY-MM-DD or YYYY-MM-DD,YYYY-MM-DD")


    queryset = queryset.distinct()
    total_count = queryset.count()
    paginated = queryset[offset:offset + limit]
    results = []

    for image in paginated:
        tags = [tag.tag.name for tag in image.image_tags.all()]

        if image.sensorbox:
            tags += [
                image.sensorbox.edge_box.plant.tenant.name,
                image.sensorbox.edge_box.edge_box_location,
                image.sensorbox.sensor_box_location,
            ]

        projects = []
        annotation_classes = set()
        for pi in image.projects.all():
            projects.append({
                "project_id": pi.project.id,
                "project_name": pi.project.name,
                "status": pi.status,
            })

            for ann in pi.annotations.all():
                annotation_classes.add(ann.annotation_class.name)

        results.append({
            "id": image.id,
            "image_id": image.image_id,
            "name": image.image_name,
            "src": image.image_file.url if image.image_file else "",
            "tags":  tags,
            "source":image.source_of_origin or "Unknown",
            "date": image.created_at.strftime("%Y-%m-%d"),
            "projects": projects,
            "annotation_classes": list(annotation_classes),
        })

    return {
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "query": query,
        "data": results
    }