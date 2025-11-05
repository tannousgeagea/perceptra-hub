

"""
FastAPI routes for dataset version management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field
import logging
from asgiref.sync import sync_to_async
from datetime import datetime

from api.dependencies import get_project_context, ProjectContext
from api.routers.versions.schemas import VersionImageAdd, VersionResponse
from projects.models import Project, ProjectImage, Version, VersionImage
from common_utils.image.utils import parse_project_image_query, apply_version_image_filters

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")

@sync_to_async
def get_dowload_url(pi:ProjectImage, expiration:int=3600):
    return pi.image.get_download_url(expiration=expiration)

@router.get(
    "/{project_id}/versions/{version_id}/images",
    summary="List Version Images"
)
async def list_version_images(
    project_id: UUID,
    version_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    split: Optional[str] = Query(None),
    q: Optional[str] = Query(
        None,
        description="Search query (e.g., 'status:annotated tag:car min-width:1920')",
        alias="q"
    ),
):
    """
    List images in dataset version.
    
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
    def get_version_images(project, version_id, skip, limit, query, split_filter):
        try:
            version = Version.objects.get(version_id=version_id, project=project)
        except Version.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
        
        queryset = VersionImage.objects.filter(version=version)
        
        if query:
            filters = parse_project_image_query(query)
            queryset = apply_version_image_filters(queryset, filters)
        
        if split_filter:
            queryset = queryset.filter(split=split_filter)
        
        total = queryset.count()
        version_images = list(
            queryset.select_related('project_image__image')[skip:skip + limit]
        )
        
        return {"total": total, "version_images": version_images}
    
    result = await get_version_images(project_ctx.project, version_id, skip, limit, q, split)
    
    return {
        "total": result["total"],
        "images": [
            {
                "id": str(vi.id),
                "project_image_id": str(vi.project_image.id),
                "image_id": str(vi.project_image.image.image_id),
                "name": vi.project_image.image.name,
                "split": vi.split,
                "annotation_count": vi.annotation_count,
                "added_at": vi.added_at.isoformat(),
                "download_url": await get_dowload_url(pi=vi.project_image)
            }
            for vi in result["version_images"]
        ]
    }