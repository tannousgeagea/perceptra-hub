import os
import math
import uuid
import time
import json
import django
import shutil
django.setup()
from django.db.models import Q, Count
from datetime import datetime, timedelta
from datetime import time as dtime
from datetime import date, timezone
from typing import Callable, Optional, Dict, AnyStr, Any
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi import status as http_status
from fastapi import Query
from pathlib import Path
from pydantic import BaseModel


from images.models import Image
from tenants.models import (
    Tenant,
    Plant,
    EdgeBox,
)
from projects.models import (
    Project,
    ProjectImage,
)

from annotations.models import (
    Annotation
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
    responses={404: {"description": "Not found"}},
)


def filter_mapping(key, value):
    try:
        if value is None:
            return None
        
        if not len(value):
            return None
        
        if value == "all":
            return None
        
        if key == "mode":
            return ("mode__mode", value)
        if key == "filename":
            return("image__image_name", value)
        if key == "classes":
            return[("annotations__annotation_class__class_id", value), ("annotations__is_active", True)]
        if key =="annotation_count":
            return("annotation_count", value)
    except Exception as err:
        raise ValueError(f"Failed to map filter value {value} filter {key}: {err}")


@router.api_route(
    "/projects/{project_id}/images", methods=["GET"], tags=["Projects"]
)
def get_project_images(
    response: Response,
    project_id:str,
    status:str,
    user_filters: Optional[str] = Query(None),
    items_per_page:int=50,
    page:int=1,
    ):
    results = {}
    try:
        filters_dict = {}
        if user_filters:
            try:
                filters_dict = json.loads(user_filters)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format for user_filters")
            
            
        print(filters_dict)
        
        
        project = Project.objects.filter(name=project_id)
        if not project:
            results['error'] = {
                'status_code': "not found",
                "status_description": f"Project {project_id} not found",
                "detail": f"Project {project_id} not found",
            }
            
            response.status_code = http_status.HTTP_404_NOT_FOUND
            return results
        
        project = project.first()
        
        if page < 1:
            page = 1    
        
        
        queryset = ProjectImage.objects.annotate(
            annotation_count=Count(
                "annotations", 
                filter=Q(annotations__is_active=True)
                )
            )
        
        lookup_filters = Q()
        lookup_filters &= Q(project=project)
        lookup_filters &= Q(status=status)
        lookup_filters &= Q(is_active=True)
        for key, value in filters_dict.items():
            filter_map = filter_mapping(key, value)
            if isinstance(filter_map, list):
                for f in filter_map:
                    lookup_filters &= Q(f)
                
            elif filter_map:
                lookup_filters &= Q(filter_map) 
        
        images = queryset.filter(lookup_filters).order_by("-added_at").distinct()
        data = []
        for image in images[(page - 1) * items_per_page:page * items_per_page]:
            annotation = Annotation.objects.filter(project_image=image, is_active=True)
            data.append(
                {
                    'project_id': image.project.name,
                    'image_id': image.image.image_id,
                    'image_name': image.image.image_name,
                    'image_url': 'http://localhost:81' + image.image.image_file.url if os.getenv('DJANGO_STORAGE') != 'azure' else image.image.image_file.url,
                    'created_at': image.image.created_at.strftime(DATETIME_FORMAT),
                    'plant': image.image.sensorbox.edge_box.plant.plant_name if image.image.sensorbox else None,
                    'edge_box': image.image.sensorbox.sensor_box_name if image.image.sensorbox else None,
                    'location': image.image.sensorbox.edge_box.edge_box_location if image.image.sensorbox else None,
                    'sub_location': image.image.sensorbox.sensor_box_location if image.image.sensorbox else None,
                    'annotated': image.annotated,
                    'reviewed': image.reviewed,
                    "annotations": [
                        {
                             "class_id": ann.annotation_class.class_id,
                             "class_name": ann.annotation_class.name,
                             "xyxyn": ann.data,
                        } for ann in annotation
                    ]
                }              
            )
            
        total_record = len(images)
        results = {
            "total_record": total_record,
            "pages": math.ceil(total_record / items_per_page),
            'unannotated': len(ProjectImage.objects.filter(project=project, status="unannotated", is_active=True)),
            'annotated': len(ProjectImage.objects.filter(project=project, status="annotated", is_active=True)),
            'reviewed': len(ProjectImage.objects.filter(project=project, status="reviewed", is_active=True)),
            "user_filters": lookup_filters,
            'data': data,
        }
    
    except HTTPException as e:
        results['error'] = {
            "status_code": "not found",
            "status_description": "Request not Found",
            "detail": f"{e}",
        }
        
        response.status_code = http_status.HTTP_404_NOT_FOUND
    
    except Exception as e:
        results['error'] = {
            'status_code': 'server-error',
            "status_description": f"Internal Server Error",
            "detail": str(e),
        }
        
        response.status_code = http_status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return results 