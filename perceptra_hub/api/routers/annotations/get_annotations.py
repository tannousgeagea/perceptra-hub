import os
import time
import django
django.setup()
from fastapi import status
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from pydantic import BaseModel
from fastapi import HTTPException
from database.models import Project
from datetime import date, timezone
from fastapi.routing import APIRoute
from fastapi import Depends, Form, Body
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, AnyStr, Any, List
from database.models import Project, Annotation
from common_utils.data.integrity import validate_project_exists
from common_utils.data.annotation.utils import load_labels


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
    prefix="/api/v1",
    tags=["Annotations"],
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)


@router.api_route(
    "/annotations/{project_name}", methods=["GET"], tags=["Annotations"]
)
def get_annotations(response: Response, project_name:str):
    results = {}
    try:
        if not validate_project_exists(project_name=project_name):
            results['error'] = {
                'status_code': 'not found',
                'status_description': f'project_name {project_name} not found',
                'detail': f'project_name {project_name} not found in db',
            }
            
            response.status_code = status.HTTP_404_NOT_FOUND
            return results
        
        project = Project.objects.get(project_name=project_name)
        annotations = Annotation.objects.filter(project=project)
        data = []
        for annotation in annotations:
            
            lb = load_labels(file=annotation.annotation_file.url)
            
            objects = []
            class_names = project.annotation_group.split('-')
            for xy in lb:
                objects.append(
                    {
                        'class_id': xy[0],
                        'class_name': class_names[int(xy[0])],
                        'xyn': [(xy[1:][i], xy[1:][i+1]) for i in range(0, len(xy[1:]), 2)],
                    }
                )
            
            data.append(
                {
                    'image': annotation.image.image_name,
                    'annotation_file': annotation.annotation_file.url,
                    'class_id': list(annotation.meta_info.get('class_id', {}).keys()),
                    'objects': objects,
                }
            )
    

        results = {
            'project_name': project_name,
            'project_type': project.project_type.project_type,
            'data': data, 
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