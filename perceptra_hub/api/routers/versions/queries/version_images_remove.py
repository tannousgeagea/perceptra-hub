

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
from api.routers.versions.schemas import VersionCreate, VersionResponse
from projects.models import Project, ProjectImage, Version, VersionImage


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")


@router.delete(
    "/{project_id}/versions/{version_id}/images",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove Images from Version"
)
async def remove_images_from_version(
    project_id: UUID,
    version_id: UUID,
    project_image_ids: List[int],
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Remove images from dataset version."""
    
    project_ctx.require_edit_permission()
    
    @sync_to_async
    def remove_images(project, version_id, project_image_ids):
        from django.db import transaction
        
        try:
            version = Version.objects.get(version_id=version_id, project=project)
        except Version.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
        
        if version.export_status in ['processing', 'completed']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot remove images from version with status '{version.export_status}'"
            )
        
        with transaction.atomic():
            deleted_count = VersionImage.objects.filter(
                version=version,
                project_image_id__in=project_image_ids
            ).delete()[0]
            
            # Update version counts
            version_images = VersionImage.objects.filter(version=version)
            version.total_images = version_images.count()
            version.total_annotations = sum(vi.annotation_count for vi in version_images)
            version.train_count = version_images.filter(split='train').count()
            version.val_count = version_images.filter(split='val').count()
            version.test_count = version_images.filter(split='test').count()
            version.save()
        
        return deleted_count
    
    deleted = await remove_images(project_ctx.project, version_id, project_image_ids)

