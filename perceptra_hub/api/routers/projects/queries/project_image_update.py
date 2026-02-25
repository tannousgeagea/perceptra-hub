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
from django.db.models import F
from api.dependencies import get_project_context, ProjectContext
from api.routers.projects.schemas import ProjectImageStatusUpdate, SplitDatasetRequest, SplitDatasetResponse
from projects.models import ProjectImage
from annotations.models import (
    Annotation,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")



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
    
    # Prevent duplicate review of same state
    if project_image.reviewed and project_image.status == 'reviewed':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image already {project_image.status}"
        )
    
    # Update review fields
    project_image.reviewed = True
    project_image.status = 'approved' if approved else 'rejected'
    project_image.reviewed_by = user
    project_image.reviewed_at = timezone.now()
    
    update_fields = ['reviewed', 'status', 'reviewed_by', 'reviewed_at', 'updated_at']
    
    # Track review history in metadata
    if not project_image.metadata:
        project_image.metadata = {}
    if 'review_history' not in project_image.metadata:
        project_image.metadata['review_history'] = []
    
    project_image.metadata['review_history'].append({
        'approved': approved,
        'reviewed_by': user.username,
        'reviewed_at': timezone.now().isoformat(),
        'feedback': feedback
    })
    
    update_fields.append('metadata')
    
    if feedback:
        project_image.feedback_provided = True
        update_fields.append('feedback_provided')
    
    # IMPORTANT: Save with update_fields to trigger signal
    project_image.save(update_fields=update_fields)
    
    # Then increment counters using F() at database level
    ProjectImage.objects.filter(id=project_image.id).update(
        review_count=F('review_count') + 1,
        last_review_version=F('last_review_version') + 1
    )
    
    # Refresh to get updated values
    project_image.refresh_from_db()
        
    # Update annotations
    Annotation.objects.filter(
        project_image=project_image,
        is_active=True
    ).update(
        reviewed=True,
        reviewed_by=user,
        reviewed_at=timezone.now()
    )
    
    return {
        "message": f"Image {'approved' if approved else 'rejected'} successfully",
        "project_image_id": str(project_image.id),
        "status": project_image.status,
        "reviewed": project_image.reviewed,
        "reviewed_by": project_image.reviewed_by.username if project_image.reviewed_by else None,
        "reviewed_at": project_image.reviewed_at.isoformat() if project_image.reviewed_at else None
    }


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
        
        # Prevent duplicate review of same state
        if project_image.reviewed and project_image.status == 'reviewed':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image already {project_image.status}"
            )
        
        # Update review fields
        project_image.reviewed = True
        project_image.status = 'approved' if approved else 'rejected'
        project_image.reviewed_by = user
        project_image.reviewed_at = timezone.now()
        
        update_fields = ['reviewed', 'status', 'reviewed_by', 'reviewed_at', 'updated_at']
        
        # Track review history in metadata
        if not project_image.metadata:
            project_image.metadata = {}
        if 'review_history' not in project_image.metadata:
            project_image.metadata['review_history'] = []
        
        project_image.metadata['review_history'].append({
            'approved': approved,
            'reviewed_by': user.username,
            'reviewed_at': timezone.now().isoformat(),
            'feedback': feedback
        })
        
        update_fields.append('metadata')
        
        if feedback:
            project_image.feedback_provided = True
            update_fields.append('feedback_provided')
        
        # IMPORTANT: Save with update_fields to trigger signal
        project_image.save(update_fields=update_fields)
        
        # Then increment counters using F() at database level
        ProjectImage.objects.filter(id=project_image.id).update(
            review_count=F('review_count') + 1,
            last_review_version=F('last_review_version') + 1
        )
        
        # Refresh to get updated values
        project_image.refresh_from_db()
            
        # Update annotations
        Annotation.objects.filter(
            project_image=project_image,
            is_active=True
        ).update(
            reviewed=True,
            reviewed_by=user,
            reviewed_at=timezone.now()
        )
        
        return {
            "message": f"Image {'approved' if approved else 'rejected'} successfully",
            "project_image_id": str(project_image.id),
            "status": project_image.status,
            "reviewed": project_image.reviewed,
            "reviewed_by": project_image.reviewed_by.username if project_image.reviewed_by else None,
            "reviewed_at": project_image.reviewed_at.isoformat() if project_image.reviewed_at else None
        }
    
    return await review_image(
        project_ctx.project,
        project_image_id,
        approved,
        feedback,
        project_ctx.user
    )
    

# ============= Bulk Project Image Review (Signal-Safe) =============
@router.patch(
    "/{project_id}/images/bulk-review",
    summary="Bulk Review Project Images",
    description="Review multiple project images while triggering model signals"
)
async def bulk_review_project_images(
    project_id: UUID,
    image_ids: List[int] = Body(..., description="List of image IDs"),
    approved: bool = Body(..., description="Approve or reject"),
    feedback: Optional[str] = Body(None, description="Optional review feedback"),
    project_ctx: ProjectContext = Depends(get_project_context)
):
    @sync_to_async
    def bulk_review(project, image_ids, approved, feedback, user):
        from django.db import transaction
        from django.utils import timezone
        from django.db.models import F

        if not image_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image IDs provided"
            )

        with transaction.atomic():

            images = (
                ProjectImage.objects
                .select_for_update()
                .filter(
                    id__in=image_ids,
                    project=project,
                    is_active=True
                )
            )

            total_requested = len(image_ids)
            processed = 0
            skipped = 0
            failed = 0

            now = timezone.now()
            status_value = "approved" if approved else "rejected"

            for img in images:
                try:
                    # Skip already reviewed images
                    if img.reviewed and img.status in ["approved", "rejected"]:
                        skipped += 1
                        continue

                    img.reviewed = True
                    img.status = status_value
                    img.reviewed_by = user
                    img.reviewed_at = now

                    if not img.metadata:
                        img.metadata = {}

                    if "review_history" not in img.metadata:
                        img.metadata["review_history"] = []

                    img.metadata["review_history"].append({
                        "approved": approved,
                        "reviewed_by": user.username,
                        "reviewed_at": now.isoformat(),
                        "feedback": feedback
                    })

                    update_fields = [
                        "reviewed",
                        "status",
                        "reviewed_by",
                        "reviewed_at",
                        "metadata",
                        "updated_at",
                    ]

                    if feedback:
                        img.feedback_provided = True
                        update_fields.append("feedback_provided")

                    # 🔥 THIS TRIGGERS post_save SIGNAL
                    img.save(update_fields=update_fields)

                    # Safe counter increment at DB level
                    ProjectImage.objects.filter(id=img.id).update(
                        review_count=F("review_count") + 1,
                        last_review_version=F("last_review_version") + 1
                    )

                    processed += 1

                except Exception:
                    failed += 1

            # 🔥 Bulk update annotations once
            if processed > 0:
                Annotation.objects.filter(
                    project_image_id__in=image_ids,
                    is_active=True
                ).update(
                    reviewed=True,
                    reviewed_by=user,
                    reviewed_at=now
                )

            return {
                "message": f"Bulk review completed",
                "total_requested": total_requested,
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
                "final_status": status_value,
            }

    return await bulk_review(
        project_ctx.project,
        image_ids,
        approved,
        feedback,
        project_ctx.user
    )


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
                ).update(
                    is_active=False, 
                    is_delete=True,
                    deleted_by=user,
                    deleted_at=timezone.now()
                )
                
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


# ============= Bulk Mark Images as Null =============
@router.patch(
    "/{project_id}/images/bulk-mark-null",
    summary="Bulk Mark Images as Null (No Objects)",
    description="Mark multiple images as null (negative samples) or unmark them"
)
async def bulk_mark_images_as_null(
    project_id: UUID,
    image_ids: List[int] = Body(..., description="List of image IDs"),
    is_null: bool = Body(..., description="True = mark as null, False = unmark"),
    reason: Optional[str] = Body(None, description="Optional reason/notes"),
    project_ctx: ProjectContext = Depends(get_project_context)
):
    @sync_to_async
    def bulk_mark_null(project, image_ids, is_null, reason, user):
        from django.utils import timezone
        from django.db import transaction

        if not image_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image IDs provided"
            )

        with transaction.atomic():

            images = (
                ProjectImage.objects
                .select_for_update()
                .filter(
                    id__in=image_ids,
                    project=project,
                    is_active=True
                )
            )

            total_requested = len(image_ids)
            processed = 0
            skipped = 0
            failed = 0
            now = timezone.now()

            for img in images:
                try:
                    if is_null:
                        # Skip if already null
                        if img.marked_as_null:
                            skipped += 1
                            continue

                        img.marked_as_null = True
                        img.status = "reviewed"
                        img.annotated = True
                        img.reviewed = True

                        # Soft delete annotations
                        deleted_count = Annotation.objects.filter(
                            project_image=img,
                            is_active=True
                        ).update(
                            is_active=False,
                            is_delete=True,
                            deleted_by=user,
                            deleted_at=now
                        )

                        if not img.metadata:
                            img.metadata = {}

                        img.metadata["null_status"] = {
                            "is_null": True,
                            "reason": reason or "No objects of interest",
                            "marked_by": user.username if user else None,
                            "marked_at": now.isoformat(),
                            "annotations_removed": deleted_count,
                        }

                    else:
                        # Skip if already not null
                        if not img.marked_as_null:
                            skipped += 1
                            continue

                        img.marked_as_null = False
                        img.status = "unannotated"
                        img.annotated = False
                        img.reviewed = False

                        if img.metadata and "null_status" in img.metadata:
                            img.metadata["null_status"]["is_null"] = False
                            img.metadata["null_status"]["unmarked_at"] = now.isoformat()
                            img.metadata["null_status"]["unmarked_by"] = (
                                user.username if user else None
                            )

                    # 🔥 Triggers post_save
                    img.save()

                    processed += 1

                except Exception:
                    failed += 1

            return {
                "message": "Bulk null operation completed",
                "total_requested": total_requested,
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
                "operation": "mark_null" if is_null else "unmark_null"
            }

    return await bulk_mark_null(
        project_ctx.project,
        image_ids,
        is_null,
        reason,
        project_ctx.user
    )

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
        project_image.save(
            update_fields=['finalized', 'status']
        )
        
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

# ============= Bulk Delete Project Images =============
@router.delete(
    "/{project_id}/images",
    status_code=status.HTTP_200_OK,
    summary="Bulk Remove Images from Project",
    description="Soft or hard delete multiple project images"
)
async def bulk_remove_images_from_project(
    project_id: UUID,
    image_ids: List[int] = Body(..., description="List of image IDs"),
    hard_delete: bool = Body(False, description="True = permanently delete"),
    project_ctx: ProjectContext = Depends(get_project_context),
):
    """
    Bulk remove images from project (soft or hard delete).
    """

    # 🔒 Require admin role
    project_ctx.require_project_role("admin", "owner")

    @sync_to_async
    def bulk_delete(project, image_ids, hard_delete):
        from django.db import transaction
        from jobs.models import JobImage

        if not image_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image IDs provided"
            )

        with transaction.atomic():

            images = (
                ProjectImage.objects
                .select_for_update()
                .filter(
                    id__in=image_ids,
                    project=project
                )
            )

            total_requested = len(image_ids)
            found_count = images.count()

            if found_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No matching project images found"
                )

            if hard_delete:
                # 🔥 HARD DELETE MODE

                # Delete annotations in bulk
                Annotation.objects.filter(
                    project_image_id__in=image_ids
                ).delete()

                # Delete job assignments in bulk
                JobImage.objects.filter(
                    project_image_id__in=image_ids
                ).delete()

                # Delete project images (triggers post_delete if exists)
                deleted_count, _ = images.delete()

                return {
                    "message": "Images permanently deleted",
                    "total_requested": total_requested,
                    "deleted": deleted_count,
                    "mode": "hard_delete"
                }

            else:
                # 🔥 SOFT DELETE MODE (Signal Safe)

                processed = 0
                skipped = 0

                for img in images:
                    if not img.is_active:
                        skipped += 1
                        continue

                    img.is_active = False

                    # 🔥 Triggers post_save
                    img.save(update_fields=["is_active", "updated_at"])

                    processed += 1

                return {
                    "message": "Images soft deleted",
                    "total_requested": total_requested,
                    "processed": processed,
                    "skipped": skipped,
                    "mode": "soft_delete"
                }

    return await bulk_delete(
        project_ctx.project,
        image_ids,
        hard_delete
    )


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
            
            return count, invalid_ids, list(project_images.values_list('id', flat=True))
    
    count, invalid_ids, finalized_ids = await batch_finalize(
        project_ctx.project,
        project_image_ids
    )
    
    # Trigger async event creation
    if finalized_ids:
        from api.tasks.activity import create_batch_finalize_events
        create_batch_finalize_events.delay(
            project_id=str(project_id),
            project_image_ids=finalized_ids,
            user_id=project_ctx.user.id,
        )
    
    return {
        "message": f"Finalized {count} images for dataset",
        "finalized_count": count,
        "invalid_ids": invalid_ids,
        "total_requested": len(project_image_ids)
    }