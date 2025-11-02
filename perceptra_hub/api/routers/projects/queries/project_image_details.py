

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
from api.routers.projects.schemas import ProjectImageOut
from projects.models import (
    ProjectImage,
)

from images.models import Image

from django.contrib.auth import get_user_model

User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects")



@router.get(
    "/{project_id}/images/{image_id}", 
    response_model=ProjectImageOut,
    summary="Get Project Image Details",
    description="Returns detailed information about a specific project image including annotations, metadata, and job assignments."
)
async def get_project_image(
    project_id: UUID,
    image_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Get detailed information about a project image including all annotations."""
    
    @sync_to_async
    def fetch_project_image(project, image_id):
        try:
            image = Image.objects.get(
                image_id=image_id,
            )
        except Image.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image {image_id} not found"
            )
        
        try:
            pi = ProjectImage.objects.select_related(
                'image',
                'mode',
                'added_by',
                'reviewed_by'
            ).prefetch_related(
                'annotations__annotation_class',
                'annotations__annotation_type',
                'job_assignments__job'
            ).get(
                image=image,
                project=project,
                is_active=True
            )
        except ProjectImage.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project image not found"
            )
        
        return {
            "id": str(pi.id),
            "image": {
                "id": str(pi.id),
                "image_id": str(image.image_id),
                "name": image.name,
                "original_filename": image.original_filename,
                "width": image.width,
                "height": image.height,
                "aspect_ratio": image.aspect_ratio,
                "file_format": image.file_format,
                "file_size": image.file_size,
                "file_size_mb": round(image.file_size_mb, 2),
                "megapixels": round(image.megapixels, 2),
                "storage_key": image.storage_key,
                "checksum": image.checksum,
                "source_of_origin": image.source_of_origin,
                "created_at": image.created_at,
                "updated_at": image.updated_at,
                "uploaded_by": image.uploaded_by.username,
                "tags": [t.name for t in image.tags.all()],
                "storage_profile": {
                    "id": str(image.storage_profile.id),
                    "name": image.storage_profile.name,
                    "backend": image.storage_profile.backend
                },
                "download_url": image.get_download_url(expiration=3600)
            },
            "status": pi.status,
            "annotated": pi.annotated,
            "reviewed": pi.reviewed,
            "finalized": pi.finalized,
            "marked_as_null": pi.marked_as_null,
            "priority": pi.priority,
            "job_assignment_status": pi.job_assignment_status,
            "mode": {
                "id": pi.mode.id,
                "mode": pi.mode.mode
            } if pi.mode else None,
            "metadata": pi.metadata,
            "added_by": pi.added_by.username if pi.added_by else None,
            "reviewed_by": pi.reviewed_by.username if pi.reviewed_by else None,
            "added_at": pi.added_at.isoformat(),
            "reviewed_at": pi.reviewed_at.isoformat() if pi.reviewed_at else None,
            "updated_at": pi.updated_at.isoformat(),
            "jobs": [
                {
                    "id": str(ja.job.id),
                    "name": ja.job.name,
                    "status": ja.job.status
                }
                for ja in pi.job_assignments.all()
            ],
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
                    "is_active": ann.is_active,
                    "created_at": ann.created_at.isoformat(),
                    "created_by": ann.created_by
                }
                for ann in pi.annotations.filter(is_active=True).select_related(
                    'annotation_class', 'annotation_type'
                )
            ]
        }
    
    return await fetch_project_image(project_ctx.project, image_id)