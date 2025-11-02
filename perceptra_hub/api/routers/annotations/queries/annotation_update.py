import os
import math
import uuid
import time
import django
import shutil
django.setup()
from django.db.models import Q
from datetime import datetime, timedelta
from datetime import time as dtime
from datetime import date, timezone
from typing import Callable, Optional, Dict, AnyStr, Any, List
from fastapi import Request
from fastapi import Response, Depends
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi import status
from pathlib import Path
from pydantic import BaseModel


from common_utils.data.annotation.raw import save_annotations

from images.models import Image
from projects.models import (
    Project,
    ProjectImage,
)

from annotations.models import (
    AnnotationType
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

class ApiRequest(BaseModel):
    project_name: str
    image_id: str
    annotation_type:str
    data: List[List]

@router.api_route(
    "/annotations", methods=["POST"], tags=["Annotations"]
)
def upload_raw_annotations(response: Response, request: ApiRequest = Depends()):
    results = {
        'status_code': 'ok',
        'status_description': '',
        'details': []
    }
    try:
        if not Project.objects.filter(name=request.project_name).exists():
            results['error'] = {
                'status_code': 'bad-request',
                'status_description': f'Project name {request.project_name} does not exist',
                'details': f'Project name {request.project_name} does not exist',
            }
            response.status_code = status.HTTP_404_NOT_FOUND
            return results
        
        if not Image.objects.filter(image_id=request.image_id).exists():
            results['error'] = {
                'status_code': 'bad-request',
                'status_description': f'Image id {request.image_id} does not exist',
                'details': f'Image image_id {request.image_id} does not exist',
            }
            response.status_code = status.HTTP_404_NOT_FOUND
            return results
        
        if not AnnotationType.objects.filter(name=request.annotation_type).exists():
            results['error'] = {
                'status_code': 'bad-request',
                'status_description': f'Annotation Type {request.annotation_type} does not exist',
                'details': f'Annotation Type {request.annotation_type} does not exist',
            }
            response.status_code = status.HTTP_404_NOT_FOUND
            return results
        
        project = Project.objects.get(name=request.project_name)
        image = Image.objects.get(image_id=request.image_id)
        annotation_type = AnnotationType.objects.get(name=request.annotation_type)
        
        project_image = ProjectImage.objects.filter(project=project, image=image)
        if project_image:
            project_image = project_image.first()
        else:
            project_image = ProjectImage.objects.create(
                image=image,
                project=project,
            )
        
        if project_image.status in ["dataset", "reviewed"]:
            return {"message": "No new annotation can be created - Image already in dataset !"}
        
        success, results = save_annotations(data=request.data, project_image=project_image, annotation_type=annotation_type)
        if success:
            project_image.annotated = True
            project_image.status = "annotated"
            project_image.save()
        
    except HTTPException as e:
        results['error'] = {
            "status_code": "not found",
            "status_description": "Request not Found",
            "detail": f"{e}",
        }
        
        response.status_code = status.HTTP_404_NOT_FOUND
    
    except Exception as e:
        results['error'] = {
            'status_code': 'server-error',
            "status_description": f"Internal Server Error",
            "detail": str(e),
        }
        
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return results