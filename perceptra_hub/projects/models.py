import uuid
from django.db import models
from images.models import (
    Image
)

from organizations.models import (
    Organization
)

from storage.models import StorageProfile

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

User = get_user_model()

class TimeStampedModel(models.Model):
    """Abstract model to track creation and modification times"""
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="%(class)s_created",
        help_text="User who created this record"
    )
    updated_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="%(class)s_updated",
        help_text="User who last updated this record"
    )

    class Meta:
        abstract = True

# Create your models here.
class ProjectType(TimeStampedModel):
    """Defines different types of computer vision projects (e.g., Object Detection, Segmentation)"""
    name = models.CharField(max_length=255, unique=True, db_index=True)
    description = models.TextField(blank=True, null=True)
    meta_info = models.JSONField(
        null=True,
        blank=True,
        help_text="Additional metadata about project type configuration"
    )
    is_active = models.BooleanField(default=True, db_index=True)
    
    class Meta:
        db_table = 'project_type'
        verbose_name = 'Project Type'
        verbose_name_plural = 'Project Types'
        ordering = ['name']
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"<ProjectType: {self.name}>"


def get_default_visibility():
    """Get or create default visibility setting"""
    visibility, _ = Visibility.objects.get_or_create(
        name='Private',
        defaults={'description': 'Only visible to organization members'}
    )
    return visibility.id


class Visibility(TimeStampedModel):
    """Defines project visibility levels (e.g., Private, Organization, Public)"""
    name = models.CharField(max_length=50, unique=True, db_index=True)
    description = models.TextField(blank=True, null=True)
    display_order = models.IntegerField(
        default=0,
        help_text="Order in which visibility options are displayed"
    )

    class Meta:
        db_table = 'visibility'
        verbose_name = 'Visibility'
        verbose_name_plural = 'Visibility Options'
        ordering = ['display_order', 'name']

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"<Visibility: {self.name}>"

class Project(TimeStampedModel):
    """
    Represents a computer vision project within an organization.
    Projects contain datasets, annotations, and trained models.
    """
    # Unique identifiers
    project_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True,
        db_index=True,
        help_text="Unique identifier for the project"
    )
    
    # Basic information
    name = models.CharField(
        max_length=255,
        help_text="Human-readable project name"
    )
    description = models.TextField(blank=True, null=True)
    thumbnail_url = models.URLField(
        blank=True,
        null=True,
        max_length=500,
        help_text="URL to project thumbnail image"
    )
    
    # Relationships
    project_type = models.ForeignKey(
        ProjectType,
        on_delete=models.PROTECT,
        related_name='projects',
        help_text="Type of computer vision task"
    )
    visibility = models.ForeignKey(
        Visibility,
        on_delete=models.PROTECT,
        default=get_default_visibility,
        related_name='projects',
        help_text="Who can view this project"
    )
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="projects",
        help_text="Organization that owns this project"
    )
    
    # Metadata
    last_edited = models.DateTimeField(auto_now=True, db_index=True)
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        help_text="Whether the project is active or archived"
    )
    is_deleted = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Soft delete flag"
    )
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="projects_deleted"
    )
    
    # Additional metadata
    settings = models.JSONField(
        default=dict,
        blank=True,
        help_text="Project-specific settings and configurations"
    )

    class Meta:
        db_table = "project"
        verbose_name = 'Project'
        verbose_name_plural = 'Projects'
        ordering = ['-last_edited', '-created_at']
        unique_together = [('organization', 'name')]
        indexes = [
            models.Index(fields=['organization', 'is_active', 'is_deleted']),
            models.Index(fields=['organization', 'project_type']),
            models.Index(fields=['visibility', 'is_active']),
        ]

    def __str__(self):
        return f"{self.name} ({self.organization.name})"
    
    def __repr__(self):
        return f"<Project: {self.project_id} - {self.name}>"
    
    def clean(self):
        """Validate model data"""
        super().clean()
        
        # Ensure project type is active
        if self.project_type and not self.project_type.is_active:
            raise ValidationError({
                'project_type': _('Cannot assign an inactive project type.')
            })
        
        # Validate that deleted projects have deletion metadata
        if self.is_deleted and not self.deleted_at:
            raise ValidationError({
                'deleted_at': _('Deleted projects must have a deletion timestamp.')
            })
    
    def save(self, *args, **kwargs):
        """Override save to run validation"""
        self.full_clean()
        super().save(*args, **kwargs)
    
    def soft_delete(self, user=None):
        """Soft delete the project"""
        from django.utils import timezone
        self.is_deleted = True
        self.is_active = False
        self.deleted_at = timezone.now()
        self.deleted_by = user
        self.save(update_fields=['is_deleted', 'is_active', 'deleted_at', 'deleted_by', 'updated_at'])
    
    def restore(self):
        """Restore a soft-deleted project"""
        self.is_deleted = False
        self.is_active = True
        self.deleted_at = None
        self.deleted_by = None
        self.save(update_fields=['is_deleted', 'is_active', 'deleted_at', 'deleted_by', 'updated_at'])
    
    @property
    def is_archived(self):
        """Check if project is archived (inactive but not deleted)"""
        return not self.is_active and not self.is_deleted
    
class ProjectMetadata(models.Model):
    project = models.ForeignKey(Project, on_delete=models.RESTRICT)
    key = models.CharField(max_length=100)
    value = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('project', 'key')
        db_table = 'project_metadata'
        verbose_name_plural = "Project Metadata"

    def __str__(self):
        return f"{self.project.name} - {self.key}: {self.value}"

class ImageMode(models.Model):
    mode = models.CharField(max_length=255)
    description = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    meta_info = models.JSONField(null=True, blank=True)
    
    class Meta:
        db_table = 'image_mode'
        verbose_name_plural = 'Image Mode'
        
    def __str__(self):
        return self.mode

class ProjectImage(models.Model):
    STATUS_CHOICES = [
        ('unannotated', 'Unannotated'),
        ('in_progress', 'In Progress'),  # Added
        ('annotated', 'Annotated'),
        ('reviewed', 'Reviewed'),
        ('approved', 'Approved'),  # Added
        ('rejected', 'Rejected'),  # Added
        ('dataset', 'Ready for Dataset'),
    ]
    
    JOB_STATUS_CHOICES = [
        ('assigned', 'Assigned to job'),
        ('waiting', 'Waiting for job'),
        ('excluded', 'Excluded from job assignment'),
    ]

    project = models.ForeignKey(
        Project, 
        on_delete=models.CASCADE, 
        related_name='project_images'
    )
    
    image = models.ForeignKey(
        Image, 
        on_delete=models.CASCADE, 
        related_name='project_assignments'
    )
    
    mode = models.ForeignKey(
        ImageMode, 
        on_delete=models.SET_NULL, 
        blank=True, 
        null=True,
        help_text=_('Optional image mode/category')
    )
    
    # Status tracking
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='unannotated',
        db_index=True
    )
    
    # Boolean flags (keep for quick filtering)
    annotated = models.BooleanField(
        default=False,
        db_index=True,
        help_text=_('Whether image has been annotated in this project')
    )
    reviewed = models.BooleanField(
        default=False,
        help_text=_('Whether annotations have been reviewed')
    )
    finalized = models.BooleanField(
        default=False,
        help_text=_('Whether image is finalized for use')
    )
    feedback_provided = models.BooleanField(
        default=False,
        help_text=_('Whether feedback has been provided')
    )
    marked_as_null = models.BooleanField(
        default=False,
        help_text=_('Whether image is marked as unusable/null')
    )
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        help_text=_('Whether this assignment is active')
    )
    
    # User tracking (NEW - IMPORTANT!)
    added_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='added_project_images',
        help_text=_('User who added image to project')
    )
    
    reviewed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reviewed_project_images',
        help_text=_('User who reviewed the annotations')
    )
    
    review_count = models.PositiveIntegerField(
        default=0,
        help_text=_('Number of times this image has been reviewed')
    )
    
    last_review_version = models.PositiveIntegerField(
        default=0,
        help_text=_('Version number of last review')
    )
    
    # Metadata (NEW - flexible additional data)
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Additional metadata (annotation stats, quality scores, etc.)')
    )
    
    # Timestamps
    added_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    reviewed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text=_('When annotations were reviewed')
    )
    
    # Priority (NEW - for annotation workflow)
    priority = models.IntegerField(
        default=0,
        help_text=_('Priority for annotation (higher = more important)')
    )

    # Job assignment tracking
    job_assignment_status = models.CharField(
        max_length=20,
        choices=JOB_STATUS_CHOICES,
        default='waiting',
        db_index=True,
        help_text=_('Job assignment status')
    )
    

    class Meta:
        unique_together = [('project', 'image')]
        db_table = 'project_image'
        verbose_name = _('Project Image')
        verbose_name_plural = _('Project Images')
        indexes = [
            models.Index(fields=['project', 'status']),
            models.Index(fields=['project', 'annotated']),
            models.Index(fields=['project', 'is_active']),
            models.Index(fields=['project', 'job_assignment_status']),
            models.Index(fields=['added_at']),
        ]
        ordering = ['-priority', '-added_at']

    def __str__(self):
        return f"{self.project.name} - {self.image.name}"
    
    def clean(self):
        """Validate project-image assignment."""
        super().clean()
        
        # Ensure image and project belong to same organization
        if self.image.organization != self.project.organization:
            raise ValidationError({
                'image': _('Image must belong to the same organization as the project')
            })
        
        # Auto-update status based on flags. 'rejected' (sent back to the
        # annotator, see reject()) is not derivable from the flags, so it is
        # kept until resubmit() or a review moves the image on.
        if self.finalized:
            self.status = 'dataset'
        elif self.reviewed:
            self.status = 'reviewed'
        elif self.status == 'rejected':
            pass
        elif self.annotated:
            self.status = 'annotated'
    
    def save(self, *args, **kwargs):
        """Override save to run validation."""
        self.full_clean()
        super().save(*args, **kwargs)
    
    # Helper methods
    def mark_as_annotated(self, user=None):
        """Mark image as annotated."""
        self.annotated = True
        self.status = 'annotated'
        if user:
            self.reviewed_by = user
        self.save()
    
    def mark_as_reviewed(self, user=None, approved=True):
        """Mark image as reviewed."""
        self.reviewed = True
        self.status = 'approved' if approved else 'rejected'
        self.reviewed_by = user
        self.reviewed_at = models.functions.Now()
        self.save()
    
    def mark_as_finalized(self):
        """Mark image as finalized for dataset."""
        self.finalized = True
        self.status = 'dataset'
        self.save()

    # ---- Null (empty / negative sample) flag ----
    # Callers own the transaction and the final save().

    def apply_null_mark(self, user=None, reason=None, now=None) -> int:
        """
        Flag the image as containing nothing to label. Active annotations are
        soft-deleted and their ids remembered, so clear_null_mark() can bring
        them back. Returns the number of annotations removed.
        """
        from django.utils import timezone
        from annotations.models import Annotation

        now = now or timezone.now()
        active = Annotation.objects.filter(project_image=self, is_active=True)
        removed_ids = list(active.values_list('id', flat=True))
        active.update(is_active=False, is_deleted=True, deleted_by=user, deleted_at=now)

        self.marked_as_null = True
        self.annotated = True   # "done": a negative sample
        self.reviewed = True
        self.status = 'reviewed'
        self.metadata = self.metadata or {}
        self.metadata['null_status'] = {
            'is_null': True,
            'reason': reason or 'No objects of interest',
            'marked_by': user.username if user else None,
            'marked_at': now.isoformat(),
            'annotations_removed': len(removed_ids),
            'removed_annotation_ids': removed_ids,
        }
        return len(removed_ids)

    def clear_null_mark(self, user=None, now=None, restore_annotations=True) -> int:
        """
        Revert a null flag. Annotations removed by apply_null_mark() are
        restored (when still soft-deleted), and the image goes back to
        'annotated' or 'unannotated' for re-review. A finalized image leaves
        the dataset, since it is no longer a confirmed sample.
        Returns the number of annotations restored.
        """
        from django.utils import timezone
        from annotations.models import Annotation

        now = now or timezone.now()
        null_status = (self.metadata or {}).get('null_status') or {}

        restored = 0
        if restore_annotations and null_status.get('removed_annotation_ids'):
            restored = Annotation.objects.filter(
                project_image=self,
                id__in=null_status['removed_annotation_ids'],
                is_deleted=True,
            ).update(is_active=True, is_deleted=False, deleted_by=None, deleted_at=None)

        has_annotations = Annotation.objects.filter(project_image=self, is_active=True).exists()

        self.marked_as_null = False
        self.finalized = False
        self.reviewed = False
        self.annotated = has_annotations
        self.status = 'annotated' if has_annotations else 'unannotated'
        if null_status:
            null_status.update({
                'is_null': False,
                'unmarked_at': now.isoformat(),
                'unmarked_by': user.username if user else None,
                'annotations_restored': restored,
            })
            null_status.pop('removed_annotation_ids', None)
            self.metadata['null_status'] = null_status
        return restored

    # ---- Review loop: send back to the annotator, resubmit ----
    # Callers own the transaction and the final save().

    def reject(self, user, reason: str, now=None) -> None:
        """
        Send the image back to its annotator with a reason. It leaves review
        (reviewed=False) and shows as 'rejected' ("needs fixes") in its job
        until resubmit() or a reviewer approves it.
        """
        from django.utils import timezone

        now = now or timezone.now()
        if self.marked_as_null:
            # "It is not empty": bring back what the null flag removed.
            self.clear_null_mark(user=user, now=now)
        self.reviewed = False
        self.finalized = False
        self.status = 'rejected'
        self.reviewed_by = user
        self.reviewed_at = now
        self.feedback_provided = True
        self.metadata = self.metadata or {}
        self.metadata['rejection'] = {
            'reason': reason,
            'rejected_by': user.username if user else None,
            'rejected_by_id': user.id if user else None,
            'rejected_at': now.isoformat(),
            'resolved': False,
        }
        self.metadata.setdefault('review_history', []).append({
            'approved': False,
            'reviewed_by': user.username if user else None,
            'reviewed_at': now.isoformat(),
            'feedback': reason,
        })

    def resubmit(self, user=None, now=None) -> None:
        """The annotator fixed a rejected image: back to review."""
        from annotations.models import Annotation

        has_annotations = Annotation.objects.filter(project_image=self, is_active=True).exists()
        self.annotated = has_annotations
        self.status = 'annotated' if has_annotations else 'unannotated'
        self.resolve_rejection(user=user, now=now, how='resubmitted')

    def resolve_rejection(self, user=None, now=None, how='approved') -> None:
        """Close an open rejection note (on resubmit or approval)."""
        from django.utils import timezone

        rejection = (self.metadata or {}).get('rejection')
        if rejection and not rejection.get('resolved'):
            rejection.update({
                'resolved': True,
                'resolved_how': how,
                'resolved_by': user.username if user else None,
                'resolved_at': (now or timezone.now()).isoformat(),
            })
            self.metadata['rejection'] = rejection

    def note_annotation_added(self) -> list:
        """
        An annotation was drawn on this image: it is annotated, and it can no
        longer be an empty (null) sample. Returns the fields that changed, for
        save(update_fields=...); empty when nothing changed.
        """
        if self.marked_as_null:
            self.marked_as_null = False
            self.reviewed = False
            self.annotated = True
            self.status = 'annotated'
            return ['marked_as_null', 'reviewed', 'annotated', 'status']
        if not self.annotated or self.status == 'unannotated':
            self.annotated = True
            self.status = 'annotated'
            return ['annotated', 'status']
        return []
    
    @property
    def annotation_progress(self) -> str:
        """Get human-readable annotation progress."""
        if self.finalized:
            return "Finalized"
        if self.reviewed:
            return "Reviewed"
        if self.annotated:
            return "Annotated"
        return "Not Annotated"
   
class ImageReview(models.Model):
    """
    One review decision on a project image: approved, or sent back to the
    annotator with a reason. Kept per decision (not just the latest state) so
    rejection rates per annotator can be computed.
    """
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='image_reviews')
    project_image = models.ForeignKey(ProjectImage, on_delete=models.CASCADE, related_name='reviews')
    job = models.ForeignKey(
        'jobs.Job', on_delete=models.SET_NULL, null=True, blank=True, related_name='image_reviews',
        help_text=_('Job the image was in when reviewed'),
    )
    reviewer = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='image_reviews_given',
    )
    annotator = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='image_reviews_received',
        help_text=_("The job's assignee when the review happened"),
    )
    approved = models.BooleanField()
    reason = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = 'image_review'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['project', 'created_at']),
            models.Index(fields=['annotator', 'approved']),
        ]

    def __str__(self):
        return f"{'approved' if self.approved else 'rejected'} {self.project_image_id} by {self.reviewer_id}"

    @classmethod
    def record(cls, project_image, reviewer, approved: bool, reason: str = ''):
        """Store one decision, attributing it to the image's current job assignee."""
        from jobs.models import JobImage

        job_image = (
            JobImage.objects.filter(project_image=project_image, is_active=True)
            .select_related('job')
            .order_by('-assigned_at')
            .first()
        )
        job = job_image.job if job_image else None
        return cls.objects.create(
            project_id=project_image.project_id,
            project_image=project_image,
            job=job,
            reviewer=reviewer,
            annotator=job.assignee if job else None,
            approved=approved,
            reason=reason or '',
        )


##############################################################################################
################################ Dataset Version #############################################
##############################################################################################
 
class DatasetSplit(models.TextChoices):
    """Dataset split types."""
    TRAIN = 'train', _('Training')
    VAL = 'val', _('Validation')
    TEST = 'test', _('Test')


class DatasetFormat(models.TextChoices):
    """Export format types."""
    YOLO = 'yolo', _('YOLO (txt)')
    COCO = 'coco', _('COCO (json)')
    PASCAL_VOC = 'pascal_voc', _('Pascal VOC (xml)')
    TFRECORD = 'tfrecord', _('TFRecord')
    CUSTOM = 'custom', _('Custom Format')


class ExportStatus(models.TextChoices):
    """Dataset export status."""
    PENDING = 'pending', _('Pending')
    PROCESSING = 'processing', _('Processing')
    COMPLETED = 'completed', _('Completed')
    FAILED = 'failed', _('Failed')
    
class Version(TimeStampedModel):
    project = models.ForeignKey(
        Project, 
        on_delete=models.CASCADE, 
        related_name='versions'
    )
    
    # Version info
    version_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True,
        db_index=True
    )
    version_name = models.CharField(max_length=100)
    version_number = models.PositiveIntegerField()  # Incremental version number
    description = models.TextField(blank=True, null=True)
    
    # Export configuration
    export_format = models.CharField(
        max_length=20,
        choices=DatasetFormat.choices,
        default=DatasetFormat.YOLO
    )
    
    # Export status
    export_status = models.CharField(
        max_length=20,
        choices=ExportStatus.choices,
        default=ExportStatus.PENDING,
        db_index=True
    )
    
    # Storage
    dataset_file = models.FileField(
        upload_to="datasets/",
        null=True,
        blank=True,
        help_text=_('Exported dataset file (zip)')
    )
    storage_profile = models.ForeignKey(
        StorageProfile,
        on_delete=models.PROTECT,  # Prevent deletion of profile with images
        related_name='dataset_versions',
        help_text=_('Storage profile where this image is stored')
    )
    storage_key = models.CharField(
        max_length=500,
        null=True,
        blank=True,
        help_text=_('Cloud storage key if uploaded to S3/Azure')
    )
    file_size = models.BigIntegerField(
        null=True,
        blank=True,
        help_text=_('Dataset file size in bytes')
    )
    
    # Statistics
    total_images = models.PositiveIntegerField(default=0)
    total_annotations = models.PositiveIntegerField(default=0)
    train_count = models.PositiveIntegerField(default=0)
    val_count = models.PositiveIntegerField(default=0)
    test_count = models.PositiveIntegerField(default=0)
    
    # Metadata
    export_config = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Export configuration (image size, augmentation, etc.)')
    )
    class_distribution = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Class distribution stats')
    )
    error_log = models.TextField(
        blank=True,
        null=True,
        help_text=_('Error log if export failed')
    )
    
    # Timestamps
    exported_at = models.DateTimeField(null=True, blank=True)
    download_count = models.PositiveIntegerField(default=0)

    class Meta:
        db_table = 'version'
        verbose_name = _('Dataset Version')
        verbose_name_plural = _('Dataset Versions')
        unique_together = ('project', 'version_number')
        ordering = ['version_number']
        indexes = [
            models.Index(fields=['project', 'export_status']),
            models.Index(fields=['version_id']),
        ]

    def __str__(self):
        return f"{self.project.name} - {self.version_name} (v{self.version_number})"

    @property
    def is_ready(self):
        """Check if dataset is ready for download."""
        return self.export_status == ExportStatus.COMPLETED and bool(self.dataset_file or self.storage_key)


    def get_download_url(self, expiration: int = 3600) -> str:
        """
        Generate a presigned download URL for this image.
        
        Args:
            expiration: URL expiration in seconds
        
        Returns:
            Presigned URL string
        """
        from storage.services import get_storage_adapter_for_profile
        
        if self.storage_profile.backend == "local":
            from api.routers.images.queries.media import _media_url
            return _media_url(
                storage_backend='local',
                storage_key=f"{self.storage_profile.config['base_path']}/{self.storage_key}"
            )

        adapter = get_storage_adapter_for_profile(self.storage_profile)
        presigned = adapter.generate_presigned_url(
            self.storage_key,
            expiration=expiration,
            method='GET'
        )
        return presigned.url


class VersionImage(models.Model):
    version = models.ForeignKey(
        Version, 
        on_delete=models.CASCADE, 
        related_name='version_images'
    )
    
    project_image = models.ForeignKey(
        ProjectImage, 
        on_delete=models.RESTRICT, 
        related_name='associated_versions'
    )
    
    # Split assignment
    split = models.CharField(
        max_length=10,
        choices=DatasetSplit.choices,
        default=DatasetSplit.TRAIN,
        db_index=True,
        help_text=_('Dataset split (train/val/test)')
    )
    
    # Metadata
    annotation_count = models.PositiveIntegerField(
        default=0,
        help_text=_('Number of annotations on this image')
    )
    included_in_export = models.BooleanField(
        default=True,
        help_text=_('Whether image was included in export')
    )
    
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('version', 'project_image')
        db_table = 'version_image'
        verbose_name_plural = 'Version Images'
        verbose_name_plural = _('Version Images')
        indexes = [
            models.Index(fields=['version', 'split']),
        ]

    def __str__(self):
        return f"{self.version.version_name} - {self.project_image.image.image_name}"
