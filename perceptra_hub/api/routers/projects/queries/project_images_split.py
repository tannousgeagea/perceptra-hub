

"""
FastAPI routes for annotation management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field
import logging
from asgiref.sync import sync_to_async
from uuid import UUID
import random

from api.dependencies import get_project_context, ProjectContext
from api.routers.projects.schemas import ProjectImageStatusUpdate, SplitDatasetRequest, SplitDatasetResponse
from projects.models import ProjectImage, ImageMode
from annotations.models import (
    Annotation,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")

# ============= Split Dataset Endpoint =============

@router.post(
    "/{project_id}/images/split",
    response_model=SplitDatasetResponse,
    summary="Split Dataset into Train/Val/Test",
    description="Split finalized images into train/validation/test sets"
)
async def split_dataset(
    project_id: UUID,
    data: SplitDatasetRequest,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """
    Split finalized images into train/val/test sets.
    
    - Only splits images with status='dataset' and finalized=True
    - Images without mode (not yet split) will be randomly assigned
    - Existing splits are preserved
    - Uses ImageMode model to track splits
    """
    
    project_ctx.require_edit_permission()
    
    @sync_to_async
    def split_images(project, data):
        from django.db import transaction
        
        # Validate ratios
        try:
            data.validate_ratios()
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Get image modes
        try:
            train_mode = ImageMode.objects.get(mode='train')
            val_mode = ImageMode.objects.get(mode='val')
            test_mode = ImageMode.objects.get(mode='test')
        except ImageMode.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Image modes not configured. Please create train/valid/test modes."
            )
        
        # Get finalized images without split (mode=None)
        images_to_split = list(
            ProjectImage.objects.filter(
                project=project,
                status='dataset',
                finalized=True,
                mode=None,
                is_active=True
            )
        )
        
        # Count already split images
        already_split = ProjectImage.objects.filter(
            project=project,
            status='dataset',
            finalized=True,
            mode__isnull=False
        ).count()
        
        if not images_to_split:
            return {
                "train_count": 0,
                "val_count": 0,
                "test_count": 0,
                "total_split": 0,
                "already_split": already_split,
                "message": "No images to split. All finalized images are already assigned to splits."
            }
        
        # Shuffle for random distribution
        random.shuffle(images_to_split)
        
        # Calculate split indices
        total = len(images_to_split)
        train_end = int(total * data.train_ratio)
        val_end = train_end + int(total * data.val_ratio)
        
        # Split images
        train_images = images_to_split[:train_end]
        val_images = images_to_split[train_end:val_end]
        test_images = images_to_split[val_end:]
        
        # Update modes in transaction
        with transaction.atomic():
            # Update train images
            if train_images:
                ProjectImage.objects.filter(
                    pk__in=[img.pk for img in train_images]
                ).update(mode=train_mode)
            
            # Update validation images
            if val_images:
                ProjectImage.objects.filter(
                    pk__in=[img.pk for img in val_images]
                ).update(mode=val_mode)
            
            # Update test images
            if test_images:
                ProjectImage.objects.filter(
                    pk__in=[img.pk for img in test_images]
                ).update(mode=test_mode)
        
        return {
            "train_count": len(train_images),
            "val_count": len(val_images),
            "test_count": len(test_images),
            "total_split": total,
            "already_split": already_split,
            "message": f"Successfully split {total} images: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}"
        }
    
    result = await split_images(project_ctx.project, data)
    
    return SplitDatasetResponse(**result)


# ============= Reset Split Endpoint =============

@router.post(
    "/{project_id}/images/reset-split",
    summary="Reset Dataset Split",
    description="Clear train/val/test assignments (set mode to None)"
)
async def reset_split(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """
    Reset split assignments for all images.
    
    - Removes mode assignment (sets to None)
    - Keeps finalized status
    - Allows re-splitting with different ratios
    """
    
    project_ctx.require_edit_permission()
    
    @sync_to_async
    def reset_image_splits(project):
        from django.db import transaction
        
        with transaction.atomic():
            updated = ProjectImage.objects.filter(
                project=project,
                status='dataset',
                finalized=True,
                mode__isnull=False
            ).update(mode=None)
        
        return {"reset_count": updated}
    
    result = await reset_image_splits(project_ctx.project)
    
    return {
        "message": f"Reset split for {result['reset_count']} images",
        "reset_count": result["reset_count"]
    }


# ============= Get Split Statistics =============

@router.get(
    "/{project_id}/images/split-stats",
    summary="Get Split Statistics",
    description="Get current train/val/test split statistics"
)
async def get_split_statistics(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Get statistics about dataset splits."""
    
    @sync_to_async
    def get_stats(project):
        from django.db.models import Count, Q
        
        # Get finalized images
        finalized = ProjectImage.objects.filter(
            project=project,
            finalized=True,
            status='dataset',
            is_active=True
        )
        
        total_finalized = finalized.count()
        
        # Count by mode
        train_count = finalized.filter(mode__mode='train').count()
        val_count = finalized.filter(mode__mode='valid').count()
        test_count = finalized.filter(mode__mode='test').count()
        unsplit_count = finalized.filter(mode__isnull=True).count()
        
        # Get images ready to finalize
        ready_to_finalize = ProjectImage.objects.filter(
            project=project,
            finalized=False,
            status__in=['annotated', 'reviewed', 'approved'],
            is_active=True
        ).count()
        
        return {
            "total_finalized": total_finalized,
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "unsplit": unsplit_count,
            "ready_to_finalize": ready_to_finalize,
            "split_ratios": {
                "train": round(train_count / total_finalized, 3) if total_finalized > 0 else 0,
                "val": round(val_count / total_finalized, 3) if total_finalized > 0 else 0,
                "test": round(test_count / total_finalized, 3) if total_finalized > 0 else 0
            }
        }
    
    return await get_stats(project_ctx.project)
