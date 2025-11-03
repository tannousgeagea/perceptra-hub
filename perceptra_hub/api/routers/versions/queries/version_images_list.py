

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


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")

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
    split: Optional[str] = Query(None)
):
    """List images in dataset version."""
    
    @sync_to_async
    def get_version_images(project, version_id, skip, limit, split_filter):
        try:
            version = Version.objects.get(version_id=version_id, project=project)
        except Version.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
        
        queryset = VersionImage.objects.filter(version=version)
        
        if split_filter:
            queryset = queryset.filter(split=split_filter)
        
        total = queryset.count()
        version_images = list(
            queryset.select_related('project_image__image')[skip:skip + limit]
        )
        
        return {"total": total, "version_images": version_images}
    
    result = await get_version_images(project_ctx.project, version_id, skip, limit, split)
    
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
                "added_at": vi.added_at.isoformat()
            }
            for vi in result["version_images"]
        ]
    }