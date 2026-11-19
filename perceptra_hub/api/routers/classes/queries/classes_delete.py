
import os
import django
django.setup()
from fastapi import APIRouter, Depends, Query
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from typing import List, Optional
import datetime
from fastapi import status as http_status
from asgiref.sync import sync_to_async
from api.dependencies import get_project_context, ProjectContext

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

@router.delete(
    "/{project_id}/classes/{class_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Annotation Class",
    description="Delete an annotation class within a project's annotation group. Prevents deletion if the class is in use by annotations."
)
async def delete_annotation_class(
    project_id: str,
    class_id: int,
    hard_delete: bool = Query(False, description="Force delete even if class is used by annotations"),
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """
    Delete an annotation class from a project’s annotation group.
    Needs the editor role or above in the project.
    """
    project_ctx.require_edit_permission()

    @sync_to_async
    def delete_class_record(project, class_id: int, hard_delete: bool):
        # Ensure annotation group exists for this project
        annotation_group = AnnotationGroup.objects.filter(project=project).first()
        if not annotation_group:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project.project_id} does not have an annotation group."
            )

        # Locate class by class_id
        annotation_class = AnnotationClass.objects.filter(
            class_id=class_id,
            annotation_group=annotation_group
        ).first()

        if not annotation_class:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Annotation class {class_id} not found in project {project.project_id}."
            )

        # Check if class is used in annotations
        annotation_count = Annotation.objects.filter(
            project_image__project=project,
            annotation_class=annotation_class,
            is_active=True
        ).count()

        if annotation_count > 0 and not hard_delete:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete class '{annotation_class.name}' — it is used by {annotation_count} active annotations."
            )

        if hard_delete and annotation_count > 0:
            Annotation.objects.filter(
                project_image__project=project,
                annotation_class=annotation_class
            ).delete()

        # Safe delete
        annotation_class.delete()
        action = "hard deleted" if hard_delete else "deleted"
        return {
            "success": True,
            "message": f"Annotation class '{annotation_class.name}' {action} successfully."
        }

    try:
        return await delete_class_record(project_ctx.project, class_id, hard_delete)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
