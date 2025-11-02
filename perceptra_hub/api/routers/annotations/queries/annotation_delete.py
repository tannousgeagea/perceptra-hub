
import os
import math
import uuid
import time
import django
import shutil
django.setup()
from uuid import UUID
from datetime import datetime, timedelta
from datetime import date, timezone
from typing import Callable, Optional, Dict, AnyStr, Any, List
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import Depends, Form, Body
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
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
    
    @sync_to_async
    def delete_annotation_record(project, annotation_id, hard_delete):
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
            annotation.is_active = False
            annotation.save(update_fields=['is_active'])
    
    await delete_annotation_record(project_ctx.project, annotation_id, hard_delete)