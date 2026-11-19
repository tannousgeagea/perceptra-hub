
import os
import math
import uuid
import django
import shutil
django.setup()
from uuid import UUID
from datetime import datetime, timedelta
from datetime import date, timezone
from typing import Optional, Dict, AnyStr, Any, List
from fastapi import APIRouter
from fastapi import Depends, Form, Body
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi import status
from pathlib import Path
from django.db import transaction
from pydantic import BaseModel
from asgiref.sync import sync_to_async

from api.dependencies import ProjectContext, get_project_context
from projects.models import (
    Project,
    ProjectImage,
)

from annotations.models import (
    Annotation,
    AnnotationClass,
    AnnotationGroup,
    AnnotationType
)

router = APIRouter(
    prefix="/projects",
)

@router.delete(
    "/{project_id}/annotations/{annotation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Annotation"
)
async def delete_annotation(
    project_id: UUID,
    annotation_id: str,
    project_ctx: ProjectContext = Depends(get_project_context),
    hard_delete: bool = False
):
    """Soft or hard delete an annotation."""
    # Annotators may remove annotations; erasing one for good is an editor's call.
    if hard_delete:
        project_ctx.require_edit_permission()
    else:
        project_ctx.require_annotate_permission()

    @sync_to_async
    def delete_annotation_record(project, annotation_id, hard_delete, user):
        try:
            annotation = Annotation.objects.get(
                annotation_uid=annotation_id,
                project_image__project=project
            )
        except Annotation.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Annotation not found"
            )
        
        if hard_delete:
            annotation.delete()
        else:
            annotation.soft_delete(user=user)
    
    await delete_annotation_record(project_ctx.project, annotation_id, hard_delete, project_ctx.user)