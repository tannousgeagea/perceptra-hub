

"""
FastAPI routes for project and job management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field
import logging
from asgiref.sync import sync_to_async

from api.dependencies import get_project_context, ProjectContext
from projects.models import (
    ProjectImage,
)

from images.models import Image

from django.contrib.auth import get_user_model

User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects")


@sync_to_async
def get_dowload_url(image:Image, expiration:int=3600):
    return image.get_download_url(expiration=expiration)


@router.get("/{project_id}/images", summary="List Project Images")
async def list_project_images(
    project_id: UUID,
    project_ctx: ProjectContext= Depends(get_project_context),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    annotated: Optional[bool] = Query(None)
):
    """List images in project."""
    
    @sync_to_async
    def get_project_images(project, skip, limit, status_filter, annotated):
        
        queryset = ProjectImage.objects.filter(
            project=project,
            is_active=True
        )
        
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        if annotated is not None:
            queryset = queryset.filter(annotated=annotated)
        
        total = queryset.count()
        project_images = list(
            queryset.select_related('image', 'mode')[skip:skip + limit]
        )
        
        return {
            "total": total,
            "images": [
                {
                    "id": str(pi.id),
                    "image_id": str(pi.image.image_id),
                    "name": pi.image.name,
                    "width": pi.image.width,
                    "height": pi.image.height,
                    "storage_key": pi.image.storage_key,
                    "download_url": pi.image.get_download_url(expiration=3600),
                    "status": pi.status,
                    "annotated": pi.annotated,
                    "reviewed": pi.reviewed,
                    "priority": pi.priority,
                    "job_assignment_status": pi.job_assignment_status,
                    "added_at": pi.added_at.isoformat(),
                    "annotations": [
                        {
                            "id": str(ann.id),
                            "annotation_uid": ann.annotation_uid,
                            "type": ann.annotation_type.name if ann.annotation_type else None,
                            "class_id": ann.annotation_class.class_id,
                            "class_name": ann.annotation_class.name,
                            "color": ann.annotation_class.color,
                            "data": ann.data,
                            "source": ann.annotation_source,
                            "confidence": ann.confidence,
                            "reviewed": ann.reviewed,
                            "is_active": ann.is_active
                        }
                        for ann in pi.annotations.filter(is_active=True).select_related(
                            'annotation_class', 'annotation_type'
                        )
                    ]
                }
                for pi in project_images
            ]
        }
    
    return await get_project_images(
        project_ctx.project, skip, limit, status, annotated
    )
