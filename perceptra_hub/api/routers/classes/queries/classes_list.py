
import os
import django
django.setup()
from fastapi import APIRouter, Depends
from fastapi import FastAPI, HTTPException, status
from fastapi import Response
from pydantic import BaseModel
from typing import Optional
from typing import List, Optional
import datetime
from uuid import UUID
from fastapi import status as http_status
from api.dependencies import ProjectContext, get_project_context
from asgiref.sync import sync_to_async

from api.routers.classes.schemas import AnnotationClassOut

# Import your Django models.
from annotations.models import (
    Annotation,
    AnnotationClass, 
    AnnotationGroup,
)

from projects.models import Project

router = APIRouter(
    prefix="/projects",
)

@router.get(
    "/{project_id}/classes",
    response_model=List[AnnotationClassOut],
    summary="List Annotation Classes for Project",
    description="Retrieve all annotation classes linked to a given project, including the count of active annotations per class.", 
)
async def get_classes(
    response:Response,
    project_id:UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    results = []
    
    @sync_to_async
    def get_annotation_classes(project:Project):
        annotation_group = AnnotationGroup.objects.filter(project=project_ctx.project).first()
        if not annotation_group:
            # A new project has no classes yet: an empty list, not an error.
            return results

        classes = annotation_group.classes.all().order_by('class_id')
        for cls in classes:
            results.append(
                AnnotationClassOut(
                    id=cls.id,
                    classId=cls.class_id,
                    name=cls.name,
                    color=cls.display_color,
                    count=Annotation.objects.filter(project_image__project=project, annotation_class=cls, is_active=True).count()
                )
            )
        return results
    
    try:
        return await get_annotation_classes(project_ctx.project)
    
    except HTTPException as e:
        raise 
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )
    
