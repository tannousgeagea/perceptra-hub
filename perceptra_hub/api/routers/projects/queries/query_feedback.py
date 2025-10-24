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
from typing import Callable, Optional, Dict, AnyStr, Any
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi import status
from pathlib import Path
from pydantic import BaseModel
import requests

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
    Annotation,
    AnnotationClass,
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



@router.api_route(
    "/projects/{project_id}/feedback", methods=["GET"], tags=["Projects"]
)
def query_tenant_feedback(
    response: Response,
    project_id:str,
    image_id:str=None,
    ):
    results = {}
    try:
        
        project = Project.objects.filter(name=project_id)
        if not project:
            results['error'] = {
                'status_code': "not found",
                "status_description": f"Project {project_id} not found",
                "detail": f"Project {project_id} not found",
            }
            
            response.status_code = status.HTTP_404_NOT_FOUND
            return results
        
        project = project.first()
        if image_id:
            images = ProjectImage.objects.filter(project=project, image__image_id=image_id, feedback_provided=False)
        else:
            images =  ProjectImage.objects.filter(project=project, feedback_provided=False)
            
        if not images:
            results['error'] = {
                'status_code': "not found",
                "status_description": f"image {image_id} for Project {project_id} not found",
                "detail": f"image {image_id} for Project {project_id} not found",
            }
            
            response.status_code = status.HTTP_404_NOT_FOUND
            return results
        
        data = []
        for image in images:

            print(image.image.image_id)
            feedback = requests.get(
                url=f'http://datahub.want:19095/api/v1/feedback/alarm/out/{image.image.image_id}'
            )

            print(feedback)
            if not feedback.status_code == 200:
                results['error'] = {
                    'status': f"{feedback.status_code}",
                    'status_description': f"{feedback.text}",
                    'detail': f"Failed to request Feedback: {feedback.reason}"
                }
                
                # response.status_code = status.HTTP_404_NOT_FOUND
                # return results
                data.append(feedback.text)
                continue
            
            is_actual_alarm = False
            feedback_json = feedback.json()['data']
            annotation = Annotation.objects.filter(project_image=image, annotation_uid__contains=image.image.image_id)
            
            if feedback_json['feedback']:
                is_actual_alarm = feedback_json['feedback']['is_actual_alarm']
                annotation_class = AnnotationClass.objects.filter(class_id=feedback_json['feedback']['rating'], annotation_group__project=project).first()
                if not annotation_class:
                    continue

                annotation.update(rating=annotation_class)
                
            if not is_actual_alarm:
                annotation.update(is_active=False)
            
            
            annotation.update(feedback_provided=True)
            image.feedback_provided = True
            image.save()
            
            data.append(feedback.json())
        
        results = {
            'data': data
        }
    
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