
import os
import time
import django
django.setup()
from fastapi import APIRouter, Depends
from fastapi import FastAPI, HTTPException, status
from fastapi import Request, Response
from pydantic import BaseModel
from fastapi.routing import APIRoute
from typing import Callable, Optional
from typing import List, Optional
import datetime
from fastapi import status as http_status
from asgiref.sync import sync_to_async

from api.routers.classes.schemas import AnnotationClassOut, AnnotationClassCreate
from api.dependencies import get_project_context, ProjectContext
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


router = APIRouter(
    prefix="/projects",
    route_class=TimedRoute,
)
@router.post(
    "/{project_id}/classes",
    response_model=AnnotationClassOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create Annotation Class",
    description="Create a new annotation class within the project's annotation group."
)
async def create_annotation_class(
    project_id:str, 
    new_class: AnnotationClassCreate,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    @sync_to_async
    def create_class_record(project: Project, data: AnnotationClassCreate):
        annotation_group = AnnotationGroup.objects.filter(project=project).first()
        if not annotation_group:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project.project_id} does not have an annotation group."
            )

        # Determine the next available class_id
        max_class = (
            AnnotationClass.objects
            .filter(annotation_group=annotation_group)
            .order_by("-class_id")
            .first()
        )
        next_class_id = (max_class.class_id + 1) if max_class else 1

        # Create new annotation class
        annotation_class = AnnotationClass.objects.create(
            annotation_group=annotation_group,
            class_id=next_class_id,
            name=data.name,
            color=data.color,
            description=data.description,
        )

        # Count existing annotations using this class
        count = Annotation.objects.filter(
            project_image__project=project,
            annotation_class=annotation_class,
            is_active=True
        ).count()
        
        return AnnotationClassOut(
            id=annotation_class.id,
            classId=annotation_class.class_id,
            name=annotation_class.name,
            color=annotation_class.color,
            count=count
        )
    
    try:
        return await create_class_record(project_ctx.project, new_class)

    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )
    
