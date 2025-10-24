import uuid
from django.db import models
from images.models import (
    Image
)

from organizations.models import (
    Organization
)

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
        
        # Auto-update status based on flags
        if self.finalized:
            self.status = 'dataset'
        elif self.reviewed:
            self.status = 'reviewed'
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
    
class Version(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='versions')
    version_name = models.CharField(max_length=100)
    version_number = models.PositiveIntegerField()  # Incremental version number
    created_at = models.DateTimeField(auto_now_add=True)
    description = models.TextField(blank=True, null=True)
    version_file = models.FileField(upload_to="versions/", null=True, blank=True)

    class Meta:
        db_table = 'version'
        verbose_name_plural = 'Versions'
        unique_together = ('project', 'version_number')
        ordering = ['version_number']

    def __str__(self):
        return f"{self.project.name} - {self.version_name} (v{self.version_number})"



class VersionImage(models.Model):
    version = models.ForeignKey(Version, on_delete=models.CASCADE, related_name='version_images')
    project_image = models.ForeignKey(ProjectImage, on_delete=models.RESTRICT, related_name='associated_versions')
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('version', 'project_image')
        db_table = 'version_image'
        verbose_name_plural = 'Version Images'

    def __str__(self):
        return f"{self.version.version_name} - {self.project_image.image.image_name}"
