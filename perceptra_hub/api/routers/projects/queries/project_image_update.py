"""
FastAPI routes for annotation management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field
import logging
from asgiref.sync import sync_to_async
import uuid

from api.dependencies import get_project_context, ProjectContext
from api.routers.projects.schemas import ProjectImageStatusUpdate, SplitDatasetRequest, SplitDatasetResponse
from projects.models import ProjectImage
from annotations.models import (
    Annotation,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")




# ============= Project Image Management Endpoints =============
@router.patch(
    "/{project_id}/images/{project_image_id}/review",
    summary="Review Project Image",
    description="Mark image as reviewed with approval/rejection"
)
async def review_project_image(
    project_id: UUID,
    project_image_id: int,
    approved: bool = Body(..., description="Whether to approve or reject"),
    feedback: Optional[str] = Body(None, description="Review feedback"),
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Review and approve/reject project image annotations."""
    
    @sync_to_async
    def review_image(project, project_image_id, approved, feedback, user):
        from django.utils import timezone
        
        try:
            project_image = ProjectImage.objects.get(
                id=project_image_id,
                project=project,
                is_active=True
            )
        except ProjectImage.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project image not found"
            )
        
        # Update review status
        project_image.reviewed = True
        project_image.status = 'approved' if approved else 'rejected'
        project_image.reviewed_by = user
        project_image.reviewed_at = timezone.now()
        
        # Add feedback to metadata
        if feedback:
            if not project_image.metadata:
                project_image.metadata = {}
            project_image.metadata['review_feedback'] = feedback
            project_image.metadata['reviewed_at'] = timezone.now().isoformat()
            project_image.feedback_provided = True
        
        project_image.save()
        
        return project_image
    
    project_image = await review_image(
        project_ctx.project,
        project_image_id,
        approved,
        feedback,
        project_ctx.user
    )
    
    return {
        "message": f"Image {'approved' if approved else 'rejected'} successfully",
        "project_image_id": str(project_image.id),
        "status": project_image.status,
        "reviewed": project_image.reviewed,
        "reviewed_by": project_image.reviewed_by.username if project_image.reviewed_by else None,
        "reviewed_at": project_image.reviewed_at.isoformat() if project_image.reviewed_at else None
    }


@router.patch(
    "/{project_id}/images/{project_image_id}/mark-null",
    summary="Mark Image as Null (No Objects)",
    description="Mark image as having no objects of interest (background/negative sample)"
)
async def mark_image_as_null(
    project_id: UUID,
    project_image_id: int,
    is_null: bool = Body(..., description="True=no objects, False=unmark"),
    reason: Optional[str] = Body(None, description="Optional reason/notes"),
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """
    Mark project image as null (no objects) or unmark it.
    
    Null images are kept active for training as negative samples.
    All existing annotations will be soft-deleted when marked as null.
    """
    
    @sync_to_async
    def mark_null(project, project_image_id, is_null, reason, user):
        from django.utils import timezone
        from django.db import transaction
        
        try:
            project_image = ProjectImage.objects.get(
                id=project_image_id,
                project=project,
                is_active=True
            )
        except ProjectImage.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project image not found"
            )
        
        with transaction.atomic():
            if is_null:
                # Mark as null (no objects)
                project_image.marked_as_null = True
                project_image.status = 'reviewed'  # Keep as annotated (negative sample)
                project_image.annotated = True  # Considered "done"
                project_image.reviewed = True
                
                # Soft delete all existing annotations (if any)
                deleted_count = Annotation.objects.filter(
                    project_image=project_image,
                    is_active=True
                ).update(is_active=False)
                
                # Store metadata
                if not project_image.metadata:
                    project_image.metadata = {}
                
                project_image.metadata['null_status'] = {
                    'is_null': True,
                    'reason': reason or 'No objects of interest',
                    'marked_by': user.username if user else None,
                    'marked_at': timezone.now().isoformat(),
                    'annotations_removed': deleted_count
                }
            else:
                # Unmark as null
                project_image.marked_as_null = False
                project_image.status = 'unannotated'
                project_image.annotated = False
                project_image.reviewed = False
                
                # Update metadata
                if project_image.metadata and 'null_status' in project_image.metadata:
                    project_image.metadata['null_status']['is_null'] = False
                    project_image.metadata['null_status']['unmarked_at'] = timezone.now().isoformat()
                    project_image.metadata['null_status']['unmarked_by'] = user.username if user else None
            
            project_image.save()
            
        return project_image
    
    project_image = await mark_null(
        project_ctx.project,
        project_image_id,
        is_null,
        reason,
        project_ctx.user
    )
    
    return {
        "message": "Image marked as null" if is_null else "Image unmarked as null",
        "project_image_id": str(project_image.id),
        "marked_as_null": project_image.marked_as_null,
        "status": project_image.status,
        "is_active": project_image.is_active,
        "metadata": project_image.metadata.get('null_status') if project_image.metadata else None
    }


@router.patch(
    "/{project_id}/images/{project_image_id}/finalize",
    summary="Finalize Image for Dataset",
    description="Mark image as ready for dataset creation"
)
async def finalize_for_dataset(
    project_id: UUID,
    project_image_id: int,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Mark project image as finalized and ready for dataset."""
    
    @sync_to_async
    def finalize_image(project, project_image_id):
        try:
            project_image = ProjectImage.objects.get(
                id=project_image_id,
                project=project,
                is_active=True
            )
        except ProjectImage.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project image not found"
            )
        
        # Validate image has annotations
        if not project_image.annotated:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot finalize image without annotations"
            )
        
        # Validate image is reviewed
        if not project_image.reviewed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot finalize image that hasn't been reviewed"
            )
        
        project_image.finalized = True
        project_image.status = 'dataset'
        project_image.save()
        
        return project_image
    
    project_image = await finalize_image(
        project_ctx.project,
        project_image_id
    )
    
    return {
        "message": "Image finalized for dataset successfully",
        "project_image_id": str(project_image.id),
        "status": project_image.status,
        "finalized": project_image.finalized
    }


@router.delete(
    "/{project_id}/images/{project_image_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove Image from Project",
    description="Remove image from project (soft delete)"
)
async def remove_image_from_project(
    project_id: UUID,
    project_image_id: int,
    project_ctx: ProjectContext = Depends(get_project_context),
    hard_delete: bool = False
):
    """Remove image from project (soft or hard delete)."""
    
    # Require admin role for deletion
    project_ctx.require_project_role('admin', 'owner')
    
    @sync_to_async
    def delete_project_image(project, project_image_id, hard_delete):
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
        
        if hard_delete:
            # Delete all annotations first
            project_image.annotations.all().delete()
            # Delete job assignments
            project_image.job_assignments.all().delete()
            # Delete project image
            project_image.delete()
        else:
            # Soft delete
            project_image.is_active = False
            project_image.save()
    
    await delete_project_image(
        project_ctx.project,
        project_image_id,
        hard_delete
    )


@router.post(
    "/{project_id}/images/batch-finalize",
    summary="Batch Finalize Images",
    description="Finalize multiple images for dataset at once"
)
async def batch_finalize_images(
    project_id: UUID,
    project_image_ids: List[int] = Body(..., description="List of project image IDs"),
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Batch finalize multiple images for dataset."""
    
    @sync_to_async
    def batch_finalize(project, image_ids):
        from django.db import transaction
        
        with transaction.atomic():
            project_images = ProjectImage.objects.filter(
                id__in=image_ids,
                project=project,
                is_active=True,
                annotated=True,
                reviewed=True
            )
            
            # Get images that don't meet criteria
            all_images = ProjectImage.objects.filter(
                id__in=image_ids,
                project=project
            )
            
            invalid_ids = []
            for img in all_images:
                if not img.annotated or not img.reviewed:
                    invalid_ids.append(img.id)
            
            # Update valid images
            count = project_images.update(
                finalized=True,
                status='dataset'
            )
            
            return count, invalid_ids
    
    count, invalid_ids = await batch_finalize(
        project_ctx.project,
        project_image_ids
    )
    
    return {
        "message": f"Finalized {count} images for dataset",
        "finalized_count": count,
        "invalid_ids": invalid_ids,
        "total_requested": len(project_image_ids)
    }