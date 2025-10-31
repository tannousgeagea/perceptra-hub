import os
import math
import uuid
import time
import django
import shutil
import logging
django.setup()
from datetime import datetime, timedelta
from datetime import date, timezone
from typing import Callable, Optional, Dict, AnyStr, Any
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import Depends, Form, Body, Query
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi import status
from fastapi import File, UploadFile
from pathlib import Path
from pydantic import BaseModel
from django.conf import settings
from common_utils.data.image.core  import save_image
from common_utils.jobs.utils import assign_uploaded_image_to_batch

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
)


class ApiRequest(BaseModel):
    image_id:Optional[str] = None
    project_id:Optional[str] = None
    source_of_origin:Optional[str] = None
    
@router.api_route(
    "/image", methods=["POST"], tags=["Images"]
)
def upload_one_image(
    response: Response, 
    file: UploadFile = File(...), 
    request: ApiRequest = Depends(),
    batch_id: Optional[str] = Query(None)
    ):
    if not file:
        results = {
            'status_code': 'bad-request',
            'status_description': f'No files included in the request',
            'details': f'No files included in the request',
        }
        
        response.status_code = status.HTTP_400_BAD_REQUEST
        return results
        
    try:
        success, results = save_image(
            file=file, 
            image_id=request.image_id,
            project_id=request.project_id,
            source=request.source_of_origin, 
            meta_info=request.model_dump()
            )

        print(results)
        # Assign to job using batch logic
        from projects.models import ProjectImage
        pi = ProjectImage.objects.filter(project__name=request.project_id, image__image_id=results['image_id']).first()
        if pi:
            assign_uploaded_image_to_batch(pi, batch_id)
    
    except Exception as e:
        results = {
            'status_code': 'server-error',
            "status_description": f"Internal Server Error",
            "detail": str(e),
        }
        
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        
    except HTTPException as e:
        results = {
            "status_code": "not found",
            "status_description": "Request not Found",
            "detail": f"{e}",
        }
        
        response.status_code = status.HTTP_404_NOT_FOUND
    
    
    return results

