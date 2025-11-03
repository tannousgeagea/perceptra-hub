

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

from common_utils.image.utils import parse_project_image_query, apply_project_image_filters
from api.routers.projects.schemas import ProjectImagesResponse
from images.models import Image
from django.contrib.auth import get_user_model

User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects")


@sync_to_async
def get_dowload_url(image:Image, expiration:int=3600):
    return image.get_download_url(expiration=expiration)


@router.get(
    "/{project_id}/images", 
    summary="List Project Images",
    response_model=ProjectImagesResponse,
)
async def list_project_images(
    project_id: UUID,
    project_ctx: ProjectContext= Depends(get_project_context),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    q: Optional[str] = Query(
        None,
        description="Search query (e.g., 'status:annotated tag:car min-width:1920')",
        alias="q"
    ),
    # Legacy filters (deprecated but supported)
    status: Optional[str] = Query(None, deprecated=True),
    annotated: Optional[bool] = Query(None, deprecated=True)
):
    """
    List project images with advanced filtering.
    
    **Query Syntax:**
    - `status:annotated|unannotated|reviewed` - Filter by status
    - `annotated:true|false` - Filter by annotated flag
    - `reviewed:true|false` - Filter by reviewed flag
    - `marked-null:true|false` - Filter null images
    - `job-status:assigned|waiting|excluded` - Filter by job assignment
    - `tag:name` - Filter by image tag
    - `filename:text` - Filter by filename
    - `min-width:1920` - Minimum width
    - `max-width:1920` - Maximum width
    - `min-height:1080` - Minimum height
    - `max-height:1080` - Maximum height
    - `min-annotations:5` - Minimum annotation count
    - `max-annotations:10` - Maximum annotation count
    - `sort:size|name|date|width|height|annotations|priority` - Sort results
    
    **Examples:**
    - `status:annotated tag:car min-width:1920`
    - `reviewed:true sort:priority`
    - `min-annotations:5 status:annotated`
    """
    
    @sync_to_async
    def get_project_images(project, skip, limit, query, legacy_status, legacy_annotated):
        
        queryset = ProjectImage.objects.filter(
            project=project,
            is_active=True
        )
        
        if query:
            filters = parse_project_image_query(query)
            queryset = apply_project_image_filters(queryset, filters)
        else:
            if legacy_status:
                queryset = queryset.filter(status=legacy_status)
            
            if legacy_annotated is not None:
                queryset = queryset.filter(annotated=legacy_annotated)
        
            queryset = queryset.order_by('-priority', '-added_at')
            
        annotated_count = queryset.filter(status='annotated').count()
        reviewed_count = queryset.filter(status='reviewed').count()
        unannotated_count = queryset.filter(status='unannotated').count()
        
        
        total = queryset.distinct().count()
        
        # Get paginated results
        project_images = list(
            queryset.select_related(
                'image',
                'image__storage_profile',
                'image__uploaded_by',
                'mode'
            ).prefetch_related(
                'image__tags',
                'annotations__annotation_class',
                'annotations__annotation_type'
            ).distinct()[skip:skip + limit]
        )
        
        return {
            "total": total,
            "annotated": annotated_count,
            "unannotated": unannotated_count,
            "reviewed": reviewed_count,
            "images": [
                {
                    "id": str(pi.id),
                    "image_id": str(pi.image.image_id),
                    "name": pi.image.name,
                    "original_filename": pi.image.original_filename,
                    "width": pi.image.width,
                    "height": pi.image.height,
                    "aspect_ratio": pi.image.aspect_ratio,
                    "file_format": pi.image.file_format,
                    "file_size": pi.image.file_size,
                    "file_size_mb": pi.image.file_size_mb,
                    "megapixels": round(pi.image.megapixels, 2),
                    "storage_key": pi.image.storage_key,
                    "checksum": pi.image.checksum,
                    "source_of_origin": pi.image.source_of_origin,
                    "created_at": pi.image.created_at,
                    "updated_at": pi.updated_at,
                    "uploaded_by": pi.image.uploaded_by.username if pi.image.uploaded_by else None,
                    "tags": [t.name for t in pi.image.tags.all()],
                    "storage_profile": {
                        "id": str(pi.image.storage_profile.id),
                        "name": pi.image.storage_profile.name,
                        "backend": pi.image.storage_profile.backend
                    },
                    "download_url": pi.image.get_download_url(expiration=3600),
                    "status": pi.status,
                    "annotated": pi.annotated,
                    "reviewed": pi.reviewed,
                    "marked_as_null": pi.marked_as_null,
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
                            "is_active": ann.is_active,
                            "created_at": ann.created_at.isoformat(),
                            "created_by": ann.created_by,
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
        project_ctx.project, skip, limit, q, status, annotated
    )
