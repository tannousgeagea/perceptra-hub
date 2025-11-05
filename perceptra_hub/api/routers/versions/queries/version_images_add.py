

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

@router.post(
    "/{project_id}/versions/{version_id}/images",
    status_code=status.HTTP_201_CREATED,
    summary="Add Images to Version"
)
async def add_images_to_version(
    project_id: UUID,
    version_id: UUID,
    data: VersionImageAdd,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Add images to dataset version with split assignment."""
    
    project_ctx.require_edit_permission()
    
    @sync_to_async
    def add_images(project, version_id, data):
        from django.db import transaction
        from annotations.models import Annotation
        
        # Get version
        try:
            version = Version.objects.get(version_id=version_id, project=project)
        except Version.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
        
        # Can't add images if export is processing or completed
        if version.export_status in ['processing', 'completed']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot add images to version with status '{version.export_status}'"
            )
        
        # Get project images
        project_images = ProjectImage.objects.filter(
            id__in=data.project_image_ids,
            project=project,
            is_active=True
        )
        
        if not project_images.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No valid project images found"
            )
        
        added = []
        
        with transaction.atomic():
            for pi in project_images:
                # Skip if already in version
                if VersionImage.objects.filter(version=version, project_image=pi).exists():
                    continue
                
                # Count annotations
                annotation_count = Annotation.objects.filter(
                    project_image=pi,
                    is_active=True
                ).count()
                
                # Create version image
                VersionImage.objects.create(
                    version=version,
                    project_image=pi,
                    split=data.split,
                    annotation_count=annotation_count
                )
                
                added.append(pi.id)
            
            # Update version counts
            version_images = VersionImage.objects.filter(version=version)
            version.total_images = version_images.count()
            version.total_annotations = sum(vi.annotation_count for vi in version_images)
            version.train_count = version_images.filter(split='train').count()
            version.val_count = version_images.filter(split='val').count()
            version.test_count = version_images.filter(split='test').count()
            version.save()
        
        return {"added": len(added), "split": data.split}
    
    result = await add_images(project_ctx.project, version_id, data)
    
    return {
        "message": f"Added {result['added']} images to version",
        "added_count": result["added"],
        "split": result["split"]
    }
