
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
from typing import Callable, Optional, Dict, AnyStr, Any
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import Depends, Form, Body
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi import status
from pathlib import Path
from asgiref.sync import sync_to_async
from api.dependencies import ProjectContext, get_project_context

from projects.models import (
    ProjectImage,
)

from annotations.models import (
    Annotation
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

@router.get(
    "/{project_id}/images/{project_image_id}/annotations",
    summary="List Annotations"
)
async def list_annotations(
    project_id: UUID,
    project_image_id: int,
    project_ctx: ProjectContext = Depends(get_project_context),
    include_inactive: bool = False
):
    """Get all annotations for a project image."""
    
    @sync_to_async
    def get_annotations(project, project_image_id, include_inactive):
        try:
            project_image = ProjectImage.objects.get(
                id=project_image_id,
                project=project
            )
        except ProjectImage.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project image not found"
            )
        
        queryset = Annotation.objects.filter(project_image=project_image)
        
        if not include_inactive:
            queryset = queryset.filter(is_active=True)
        
        annotations = list(
            queryset.select_related(
                'annotation_type',
                'annotation_class'
            ).order_by('-created_at')
        )
        
        return annotations
    
    annotations = await get_annotations(
        project_ctx.project,
        project_image_id,
        include_inactive
    )
    
    return {
        "count": len(annotations),
        "annotations": [
            {
                "id": str(ann.id),
                "annotation_uid": ann.annotation_uid,
                "type": ann.annotation_type.name if ann.annotation_type else None,
                "class_id": ann.annotation_class.class_id,
                "class_name": ann.annotation_class.name,
                "color": ann.annotation_class.color,
                "data": {
                    "id": ann.annotation_uid,
                    "x": ann.data[0],
                    "y": ann.data[1],
                    "width": ann.data[2] - ann.data[0],
                    "height": ann.data[3] - ann.data[1],
                    "label": ann.annotation_class.name,
                    "color": ann.annotation_class.color
                },
                "source": ann.annotation_source,
                "confidence": ann.confidence,
                "reviewed": ann.reviewed,
                "is_active": ann.is_active,
                "created_at": ann.created_at.isoformat(),
                "created_by": ann.created_by
            }
            for ann in annotations
        ]
    }