
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
from typing import List, Optional
import datetime
from fastapi import status as http_status

# Import your Django models.
from annotations.models import (
    Annotation,
    AnnotationClass, 
    AnnotationGroup,
)

from projects.models import Project

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


class AnnotationClassOut(BaseModel):
    id: int
    name: str
    color: str
    count: int

class AnnotationClassCreate(BaseModel):
    name: str
    color: str
    count: Optional[int] = 0
    description: Optional[str] = None

router = APIRouter(
    route_class=TimedRoute,
)
@router.api_route(
    "/classes", methods=["POST"], tags=["Annotation Classes"])
def update_class(project_id:str, new_class: AnnotationClassCreate) -> AnnotationClassOut:
    try:
        project = Project.objects.filter(name=project_id).first()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found."
            )
        
        annotation_group = AnnotationGroup.objects.filter(project=project).first()
        if not annotation_group:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{project_id} has not annotation group."
            )
        
        max_class_id = AnnotationClass.objects.filter(
            annotation_group=annotation_group
            ).order_by('-class_id').first().class_id or 0

        annotation_class = AnnotationClass.objects.create(
            annotation_group=annotation_group,
            class_id=max_class_id + 1,
            name=new_class.name,
            color=new_class.color,
            description=new_class.description
        )

        return AnnotationClassOut(
            id=annotation_class.id,
            name=annotation_class.name,
            color=annotation_class.color,
            count=new_class.count
        )
    
    except HTTPException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unexpected Error: {e}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )
    
