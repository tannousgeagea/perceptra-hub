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
from fastapi import File, UploadFile
from fastapi import Depends, Form, Body
from datetime import datetime, timedelta
from common_utils.data.annotation.core import save_annotation
from common_utils.data.annotation.raw import save_raw_annotation
from typing import Callable, Optional, Dict, AnyStr, Any, List

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


class ApiRequest(BaseModel):
    project_name: str = File(...)
    
    
@router.api_route(
    "/annotations", methods=["POST"], tags=["Annotations"]
)
def upload_annotations(response: Response, files: list[UploadFile] = File(...), request: ApiRequest = Depends()):
    results = {
        'status_code': 'ok',
        'status_description': '',
        'details': []
    }
    
    try:
        if not files:
            results['error'] = {
                'status_code': 'bad-request',
                'status_description': f'No files included in the request',
                'details': f'No files included in the request',
            }
            response.status_code = status.HTTP_400_BAD_REQUEST
            return results
            
        if not Project.objects.filter(project_name=request.project_name).exists():
            results['error'] = {
                'status_code': 'bad-request',
                'status_description': f'Project name {request.project_name} does not exist',
                'details': f'Project name {request.project_name} does not exist',
            }
            response.status_code = status.HTTP_400_BAD_REQUEST
            return results
        
        failed_annotations = []
        saved_annotations = []
        for file in files:
            success, result = save_annotation(file=file, project_name=request.project_name)
            if not success:
                failed_annotations.append(result)
                continue
            
            results['details'].append(result)
            saved_annotations.append(file.filename)
        
        if failed_annotations:
            results['status_code'] = 'partial-success'
            results['status_description'] = f'{len(saved_annotations)} images uploaded successfully, {len(failed_annotations)} images failed'
            results['details'].extend(failed_annotations)
        else:
            results['status_description'] = f'{len(saved_annotations)} images uploaded successfully'
        
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



class ApiRequest(BaseModel):
    project_name: str
    data: List[Dict] 

@router.api_route(
    "/annotations/raw", methods=["POST"], tags=["Annotations"]
)
def upload_raw_annotations(response: Response, request: ApiRequest = Depends()):
    results = {
        'status_code': 'ok',
        'status_description': '',
        'details': []
    }
    try:    
        
        if not Project.objects.filter(project_name=request.project_name).exists():
            results['error'] = {
                'status_code': 'bad-request',
                'status_description': f'Project name {request.project_name} does not exist',
                'details': f'Project name {request.project_name} does not exist',
            }
            response.status_code = status.HTTP_400_BAD_REQUEST
            return results
        
        project = Project.objects.get(project_name=request.project_name)
        
        failed_annotations = []
        saved_annotations = []
        for data in request.data:            
            success, result = save_raw_annotation(data=data, project=project)
            
            if not success:
                failed_annotations.append(result)
                continue
            
            results['details'].append(result)
            saved_annotations.append(data['filename'])
        
        if failed_annotations:
            results['status_code'] = 'partial-success'
            results['status_description'] = f'{len(saved_annotations)} images uploaded successfully, {len(failed_annotations)} images failed'
            results['details'].extend(failed_annotations)
        else:
            results['status_description'] = f'{len(saved_annotations)} images uploaded successfully'
        
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