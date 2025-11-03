
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
from api.routers.versions.schemas import VersionStatistics
from projects.models import Project, ProjectImage, Version, VersionImage


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")



@router.get(
    "/{project_id}/versions/{version_id}/statistics",
    response_model=VersionStatistics,
    summary="Get Version Statistics"
)
async def get_version_statistics(
    project_id: UUID,
    version_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Get detailed statistics for dataset version."""
    
    @sync_to_async
    def get_stats(project, version_id):
        from django.db.models import Count
        from annotations.models import Annotation
        
        try:
            version = Version.objects.get(version_id=version_id, project=project)
        except Version.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
        
        # Get version images
        version_images = VersionImage.objects.filter(version=version)
        project_image_ids = list(version_images.values_list('project_image_id', flat=True))
        
        # Get class distribution
        class_dist = Annotation.objects.filter(
            project_image_id__in=project_image_ids,
            is_active=True
        ).values(
            'annotation_class__name'
        ).annotate(
            count=Count('id')
        ).order_by('-count')
        
        class_distribution = {
            item['annotation_class__name']: item['count']
            for item in class_dist
        }
        
        # Calculate average
        avg_annotations = version.total_annotations / version.total_images if version.total_images > 0 else 0
        
        return {
            "total_images": version.total_images,
            "total_annotations": version.total_annotations,
            "splits": {
                "train": version.train_count,
                "val": version.val_count,
                "test": version.test_count
            },
            "class_distribution": class_distribution,
            "average_annotations_per_image": round(avg_annotations, 2)
        }
    
    return await get_stats(project_ctx.project, version_id)
