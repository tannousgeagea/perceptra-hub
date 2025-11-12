# apps/activity/signals.py
from django.db.models.signals import post_save, pre_save, post_delete
from django.dispatch import receiver
from django.db import transaction
from images.models import Image
from projects.models import ProjectImage
from annotations.models import Annotation
from .models import ActivityEvent, ActivityEventType
from api.tasks.activity import update_user_metrics_async
import time


class ActivityTracker:
    """Context manager for tracking action duration."""
    
    def __init__(self):
        self.start_time = None
        self.duration_ms = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.duration_ms = int((time.time() - self.start_time) * 1000)


def create_activity_event(
    organization,
    event_type,
    user=None,
    project=None,
    metadata=None,
    session_id=None,
    duration_ms=None
):
    """
    Factory function for creating activity events.
    
    Usage:
        create_activity_event(
            organization=image.organization,
            event_type=ActivityEventType.IMAGE_UPLOAD,
            user=request.user,
            metadata={'image_id': str(image.image_id), 'file_size_mb': image.file_size_mb}
        )
    """
    return ActivityEvent.objects.create(
        organization=organization,
        event_type=event_type,
        user=user,
        project=project,
        metadata=metadata or {},
        session_id=session_id,
        duration_ms=duration_ms
    )


# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

@receiver(post_save, sender=Image)
def track_image_upload(sender, instance, created, **kwargs):
    """Track when images are uploaded."""
    if created:
        create_activity_event(
            organization=instance.organization,
            event_type=ActivityEventType.IMAGE_UPLOAD,
            user=instance.uploaded_by,
            metadata={
                'image_id': str(instance.image_id),
                'file_size_mb': float(instance.file_size_mb),
                'file_format': instance.file_format,
                'source': instance.source_of_origin,
            }
        )

        # Trigger async metric update
        transaction.on_commit(lambda: update_user_metrics_async.delay(
            user_id=instance.uploaded_by_id,
            organization_id=instance.organization_id,
            event_type=ActivityEventType.IMAGE_UPLOAD,
        ))



@receiver(post_save, sender=ProjectImage)
def track_image_project_assignment(sender, instance, created, **kwargs):
    """Track when images are added to projects."""
    if created:
        create_activity_event(
            organization=instance.project.organization,
            event_type=ActivityEventType.IMAGE_ADD_TO_PROJECT,
            user=instance.added_by,
            project=instance.project,
            metadata={
                'project_image_id': instance.id,
                'image_id': str(instance.image.image_id),
                'status': instance.status,
            }
        )

        # Trigger async metric update
        transaction.on_commit(lambda: update_user_metrics_async.delay(
            user_id=instance.added_by_id or instance.added_by_id,
            organization_id=instance.project.organization_id,
            event_type=ActivityEventType.IMAGE_ADD_TO_PROJECT,
            project_id=instance.project.id,
        ))


@receiver(post_save, sender=Annotation)
def track_annotation_activity(sender, instance, created, **kwargs):
    """Track annotation creation and updates."""
    
    # Determine event type
    if created:
        event_type = ActivityEventType.ANNOTATION_CREATE
    elif instance.is_deleted:
        event_type = ActivityEventType.ANNOTATION_DELETE
    else:
        event_type = ActivityEventType.ANNOTATION_UPDATE
    
    # Build metadata
    metadata = {
        'annotation_id': instance.annotation_uid,
        'annotation_class': instance.annotation_class.name,
        'annotation_source': instance.annotation_source,  # 'manual' or 'prediction'
        'project_image_id': instance.project_image.id,
        'image_id': str(instance.project_image.image.image_id),
    }
    
    # Add prediction-specific tracking
    if instance.annotation_source == 'prediction':
        metadata.update({
            'confidence': float(instance.confidence) if instance.confidence else None,
            'edit_type': instance.edit_type,
            'edit_magnitude': float(instance.edit_magnitude) if instance.edit_magnitude else None,
        })
        
        # Track if prediction was edited
        if not created and instance.version > 1:
            if instance.edit_type == Annotation.EditType.CLASS_CHANGE:
                event_type = ActivityEventType.PREDICTION_EDIT
            elif instance.edit_type in [Annotation.EditType.MAJOR, Annotation.EditType.MINOR]:
                event_type = ActivityEventType.PREDICTION_EDIT
    
    create_activity_event(
        organization=instance.project_image.project.organization,
        event_type=event_type,
        user=instance.updated_by or instance.created_by,
        project=instance.project_image.project,
        metadata=metadata
    )
    
    # Trigger async metric update
    transaction.on_commit(lambda: update_user_metrics_async.delay(
        user_id=instance.updated_by_id or instance.created_by_id,
        organization_id=instance.project_image.project.organization_id,
        event_type=event_type,
        project_id=instance.project_image.project.id,
    ))


@receiver(post_save, sender=ProjectImage)
def track_image_status_changes(sender, instance, created, update_fields, **kwargs):
    """Track review, approval, and finalization."""
    if created or not update_fields:
        return
    
    # Check if status changed
    if 'status' in update_fields or 'reviewed' in update_fields:
        
        # Image reviewed
        if instance.reviewed and 'reviewed' in update_fields:
            event_type = ActivityEventType.IMAGE_REVIEW
            create_activity_event(
                organization=instance.project.organization,
                event_type=ActivityEventType.IMAGE_REVIEW,
                user=instance.reviewed_by,
                project=instance.project,
                metadata={
                    'project_image_id': instance.id,
                    'status': instance.status,
                    'approved': instance.status == 'approved',
                }
            )

            # Trigger async metric update
            transaction.on_commit(lambda: update_user_metrics_async.delay(
                user_id=instance.reviewed_by_id or instance.added_by_id,
                organization_id=instance.project.organization_id,
                event_type=event_type,
                project_id=instance.project.id,
            ))
        
        # Image finalized
        if instance.finalized and 'finalized' in update_fields:
            event_type = ActivityEventType.IMAGE_FINALIZE
            create_activity_event(
                organization=instance.project.organization,
                event_type=ActivityEventType.IMAGE_FINALIZE,
                user=instance.reviewed_by,
                project=instance.project,
                metadata={
                    'project_image_id': instance.id,
                }
            )
            
            # Trigger async metric update
            transaction.on_commit(lambda: update_user_metrics_async.delay(
                user_id=instance.reviewed_by_id or instance.added_by_id,
                organization_id=instance.project.organization_id,
                event_type=event_type,
                project_id=instance.project.id,
            ))

@receiver(post_delete, sender=ProjectImage)
def track_image_project_removal(sender, instance, **kwargs):
    """Track when images are removed from projects."""
    create_activity_event(
        organization=instance.project.organization,
        event_type=ActivityEventType.IMAGE_REMOVE_FROM_PROJECT,
        user=instance.reviewed_by,  # or get from request context
        project=instance.project,
        metadata={
            'project_image_id': instance.id,
            'image_id': str(instance.image.image_id),
            'status': instance.status,
            'was_annotated': instance.annotated,
            'was_reviewed': instance.reviewed,
        }
    )
    
@receiver(post_save, sender=ProjectImage)
def track_image_project_removal(sender, instance, created, update_fields, **kwargs):
    """Track when images are removed from projects (soft delete)."""
    if not created and update_fields and 'is_active' in update_fields:
        if not instance.is_active:  # Image was deactivated
            create_activity_event(
                organization=instance.project.organization,
                event_type=ActivityEventType.IMAGE_REMOVE_FROM_PROJECT,
                user=instance.reviewed_by,  # Set this in your view
                project=instance.project,
                metadata={
                    'project_image_id': instance.id,
                    'image_id': str(instance.image.image_id),
                    'status': instance.status,
                }
            )
