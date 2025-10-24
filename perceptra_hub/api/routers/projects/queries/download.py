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

from images.models import Image
from projects.models import (
    Project,
    ProjectImage
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


def xyxy2xywh(xyxy):
    """
    Convert bounding box coordinates from (xmin, ymin, xmax, ymax) format to (x, y, width, height) format.

    Parameters:
    - xyxy (Tuple[int, int, int, int]): A tuple representing the bounding box coordinates in (xmin, ymin, xmax, ymax) format.

    Returns:
    - Tuple[int, int, int, int]: A tuple representing the bounding box in (x, y, width, height) format. 
                                 (x, y) are  the center of the bounding box.
    """
    xmin, ymin, xmax, ymax = xyxy
    w = xmax - xmin
    h = ymax - ymin
    return (xmin + w/2, ymin + h/2, w, h)


@router.api_route(
    "/projects/{project_id}/download", methods=["GET"], tags=["Projects"]
)
def get_project_images(
    response: Response,
    project_id:str,
    mode:str="train"
    ):
    results = {}
    try:
        project = Project.objects.filter(name=project_id)
        if not project:
            results['error'] = {
                'status_code': 'bad-request',
                'status_description': f'Project name {project_id} does not exist',
                'details': f'Project name {project_id} does not exist',
            }
            response.status_code = status.HTTP_404_NOT_FOUND
            return results
        
        project = project.first()
        images = ProjectImage.objects.filter(project=project, reviewed=True, mode__mode=mode)
        if not images.exists():
            raise HTTPException(status_code=404, detail="No images found for this project")
        
        save_location = '/media/download'
        save_location = os.path.join(save_location, f"{project.name.replace(' ', '_')}")
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        
        for image in images:
            annotations = Annotation.objects.filter(
                project_image=image,
                is_active=True
                )
            
            file_name = f"{image.image.image_name}.txt" 
            image_location = os.path.join(save_location, mode, 'images')
            label_location = os.path.join(save_location, mode, "labels")
            
            os.makedirs(image_location, exist_ok=True)
            os.makedirs(label_location, exist_ok=True)
            
            if not annotations.exists():
                shutil.copy(image.image.image_file.url, image_location)
                with open(label_location + "/" + file_name, "w") as file:
                    file.writelines([])
                    
                continue
            
            data = [
                [0] + list(xyxy2xywh(ann.data)) for ann in annotations
            ]
            
            lines = (("%g " * len(line)).rstrip() % tuple(line) + "\n" for line in data)

            
            shutil.copy(image.image.image_file.url, image_location)
            with open(label_location + "/" + file_name, "w") as file:
                file.writelines(lines)
            
        
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
