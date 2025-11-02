
import uuid
from django.db import models
from django.utils import timezone
from projects.models import Project, ProjectImage
from django.contrib.auth import get_user_model
User = get_user_model()

class ActiveJobManager(models.Manager):
    """Manager that filters out soft-deleted jobs by default."""
    def get_queryset(self):
        return super().get_queryset().filter(is_active=True)

class ActiveJobImageManager(models.Manager):
    """Manager that filters out soft-deleted job-image links."""
    def get_queryset(self):
        return super().get_queryset().filter(is_active=True)



class JobStatus(models.TextChoices):
    UNASSIGNED = "unassigned", "Unassigned"
    ASSIGNED = "assigned", "Assigned"
    IN_REVIEW = "in_review", "In Review"
    COMPLETED = "completed", "Completed"
    SLICED = "sliced", "Sliced"

class Job(models.Model):
    job_id = models.UUIDField(
        default=uuid.uuid4, 
        editable=False,
        db_index=True,
        help_text="Unique Identifier for job"
    )
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="jobs")
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    assignee = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="assigned_jobs")
    
    status = models.CharField(max_length=20, choices=JobStatus.choices, default=JobStatus.UNASSIGNED)
    image_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    batch_id = models.CharField(max_length=255, null=True, blank=True)

    # 👇 audit fields
    created_by = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="created_jobs",
    )
    updated_by = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="updated_jobs",
    )

    # 👇 soft delete
    is_active = models.BooleanField(default=True, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    objects = models.Manager()  # default manager
    active = ActiveJobManager()  # only non-deleted jobs

    class Meta:
        db_table = "job"
        verbose_name_plural = "Jobs"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.project.name} - {self.name}"

    def soft_delete(self, user=None):
        """Soft-delete the job without removing it from DB."""
        self.is_active = False
        self.deleted_at = timezone.now()
        if user:
            self.updated_by = user
        self.save(update_fields=["is_active", "deleted_at", "updated_by", "updated_at"])

class JobImage(models.Model):
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="images")
    job_image_id = models.UUIDField(
        default=uuid.uuid4, 
        editable=False,
        db_index=True,
        help_text="Unique Identifier for job"
    )
    project_image = models.ForeignKey(ProjectImage, on_delete=models.CASCADE, related_name="job_assignments")
    created_at = models.DateTimeField(auto_now_add=True)

    # Tracking and state
    assigned_at = models.DateTimeField(auto_now_add=True)
    unassigned_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    # Audit fields
    created_by = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="created_job_images",
    )
    updated_by = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="updated_job_images",
    )

    objects = models.Manager()      # Default manager
    active = ActiveJobImageManager()  # Filtered manager
    
    class Meta:
        unique_together = ("job", "project_image")
        db_table = "job_image"
        verbose_name_plural = "Job Images"

    def __str__(self):
        return f"{self.job.name} - {self.project_image}"

    def soft_delete(self, user=None):
        """Soft-delete the job-image link (unassign)."""
        self.is_active = False
        self.deleted_at = timezone.now()
        self.unassigned_at = timezone.now()
        if user:
            self.updated_by = user
        self.save(
            update_fields=[
                "is_active",
                "deleted_at",
                "unassigned_at",
                "updated_by",
                "updated_at" if hasattr(self, "updated_at") else None,
            ]
        )