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
from projects.models import ImageReview, ProjectImage
from annotations.models import (
    Annotation,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")



REJECT_FIELDS = [
    'reviewed', 'finalized', 'status', 'reviewed_by', 'reviewed_at', 'feedback_provided',
    'metadata', 'marked_as_null', 'annotated', 'updated_at',
]


def _apply_review(project_image, approved: bool, feedback: Optional[str], user, now) -> None:
    """
    One review decision on one image (caller holds the transaction).

    Approve: the image is reviewed, its annotations are marked reviewed, and
    an open rejection note is closed. Reject: the image goes back to its job's
    annotator as 'rejected' with the reason; annotations are left unreviewed.
    Every decision is stored as an ImageReview.
    """
    if approved:
        project_image.reviewed = True
        project_image.status = 'reviewed'
        project_image.reviewed_by = user
        project_image.reviewed_at = now
        project_image.metadata = project_image.metadata or {}
        project_image.metadata.setdefault('review_history', []).append({
            'approved': True,
            'reviewed_by': user.username,
            'reviewed_at': now.isoformat(),
            'feedback': feedback,
        })
        project_image.resolve_rejection(user=user, now=now, how='approved')
        update_fields = ['reviewed', 'status', 'reviewed_by', 'reviewed_at', 'metadata', 'updated_at']
        if feedback:
            project_image.feedback_provided = True
            update_fields.append('feedback_provided')
        project_image.save(update_fields=update_fields)
        Annotation.objects.filter(project_image=project_image, is_active=True).update(
            reviewed=True, reviewed_by=user, reviewed_at=now
        )
    else:
        project_image.reject(user=user, reason=feedback, now=now)
        project_image.save(update_fields=REJECT_FIELDS)

    ProjectImage.objects.filter(id=project_image.id).update(
        review_count=F('review_count') + 1,
        last_review_version=F('last_review_version') + 1
    )
    ImageReview.record(project_image, reviewer=user, approved=approved, reason=feedback or '')


def _check_review_request(project_ctx: ProjectContext, approved: bool, feedback: Optional[str]) -> Optional[str]:
    """Rejecting needs review permission and a reason; returns the cleaned reason."""
    feedback = (feedback or '').strip() or None
    if not approved:
        project_ctx.require_review_permission()
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A reason is required to send an image back",
            )
    return feedback


@sync_to_async
def review_image(project, project_image_id, approved, feedback, user):
    from django.db import transaction
    from django.utils import timezone

    with transaction.atomic():
        try:
            project_image = ProjectImage.objects.select_for_update().get(
                id=project_image_id,
                project=project,
                is_active=True
            )
        except ProjectImage.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project image not found"
            )

        if project_image.finalized:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image is already in a dataset"
            )
        # Approving twice is a no-op error; rejecting a reviewed image is the point.
        if approved and project_image.reviewed and project_image.status == 'reviewed':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image already {project_image.status}"
            )

        _apply_review(project_image, approved, feedback, user, timezone.now())

    project_image.refresh_from_db()
    return {
        "message": "Image approved" if approved else "Image sent back to the annotator",
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
    """
    Approve an image, or send it back to its annotator (approved=false) with
    the reason in `feedback`. Sending back needs review permission.
    """
    feedback = _check_review_request(project_ctx, approved, feedback)

    return await review_image(
        project_ctx.project,
        project_image_id,
        approved,
        feedback,
        project_ctx.user
    )
    

@router.post(
    "/{project_id}/images/{project_image_id}/resubmit",
    summary="Resubmit a Rejected Image",
    description="The annotator fixed an image that was sent back: return it to review"
)
async def resubmit_project_image(
    project_id: UUID,
    project_image_id: int,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    project_ctx.require_annotate_permission()

    @sync_to_async
    def resubmit(project, user):
        from django.db import transaction

        with transaction.atomic():
            try:
                project_image = ProjectImage.objects.select_for_update().get(
                    id=project_image_id, project=project, is_active=True
                )
            except ProjectImage.DoesNotExist:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project image not found")
            if project_image.status != 'rejected':
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only an image that was sent back can be resubmitted",
                )
            project_image.resubmit(user=user)
            project_image.save(update_fields=['annotated', 'status', 'metadata', 'updated_at'])
        return project_image

    project_image = await resubmit(project_ctx.project, project_ctx.user)
    return {
        "message": "Image sent back for review",
        "project_image_id": str(project_image.id),
        "status": project_image.status,
    }


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
    feedback = _check_review_request(project_ctx, approved, feedback)

    @sync_to_async
    def bulk_review(project, image_ids, approved, feedback, user):
        from django.db import transaction
        from django.utils import timezone

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
            status_value = "reviewed" if approved else "rejected"

            for img in images:
                # Nothing to do: already in that state, or already in a dataset.
                if img.finalized or (approved and img.reviewed) or (not approved and img.status == "rejected"):
                    skipped += 1
                    continue
                try:
                    # One savepoint per image, so a failure does not poison the batch.
                    with transaction.atomic():
                        _apply_review(img, approved, feedback, user, now)
                    processed += 1
                except Exception:
                    logger.exception("Bulk review failed for project image %s", img.id)
                    failed += 1

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
                project_image.apply_null_mark(user=user, reason=reason)
            elif project_image.marked_as_null:
                # Unmarking a non-null image is a no-op: it must not reset
                # a normally annotated / reviewed image.
                project_image.clear_null_mark(user=user)
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
                        if img.marked_as_null:
                            skipped += 1
                            continue
                        img.apply_null_mark(user=user, reason=reason, now=now)
                    else:
                        if not img.marked_as_null:
                            skipped += 1
                            continue
                        img.clear_null_mark(user=user, now=now)

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