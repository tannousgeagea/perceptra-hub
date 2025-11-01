
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

from api.dependencies import ProjectContext, get_project_context
from api.routers.classes.schemas import AnnotationClassUpdate, AnnotationClassOut
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
@router.patch(
    "/{project_id}/classes/{class_id}",
    response_model=AnnotationClassOut,
    summary="Update Annotation Class",
    description="Update name, color, or description of an annotation class within the project."
)
async def update_class(
    project_id: str,
    class_id: int,
    updates: AnnotationClassUpdate,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """
    Update an existing annotation class belonging to a project's annotation group.
    """

    @sync_to_async
    def update_class_record(project, class_id: int, updates: AnnotationClassUpdate):
        # Ensure annotation group exists for this project
        annotation_group = AnnotationGroup.objects.filter(project=project).first()
        if not annotation_group:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project.project_id} does not have an annotation group."
            )

        # Find the class within this project’s annotation group
        annotation_class = AnnotationClass.objects.filter(
            class_id=class_id, annotation_group=annotation_group
        ).first()

        if not annotation_class:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Annotation class {class_id} not found in project {project.project_id}."
            )

        # Apply updates
        for key, value in updates.model_dump(exclude_unset=True).items():
            setattr(annotation_class, key, value)
        annotation_class.save()

        # Count current annotations linked to this class
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
        return await update_class_record(project_ctx.project, class_id, updates)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
