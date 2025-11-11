# apps/activity/models.py
from django.db import models
from django.contrib.postgres.indexes import BTreeIndex
from django.contrib.auth import get_user_model
from organizations.models import Organization
from projects.models import Project
import uuid

User = get_user_model()


class ActivityEventType(models.TextChoices):
    """Categorized event types for efficient filtering."""
    # Image Management
    IMAGE_UPLOAD = 'image_upload', 'Image Uploaded'
    IMAGE_ADD_TO_PROJECT = 'image_add_project', 'Image Added to Project'
    IMAGE_REMOVE_FROM_PROJECT = 'image_remove_project', 'Image Removed from Project'
    
    # Annotation Events
    ANNOTATION_CREATE = 'annotation_create', 'Annotation Created'
    ANNOTATION_UPDATE = 'annotation_update', 'Annotation Updated'
    ANNOTATION_DELETE = 'annotation_delete', 'Annotation Deleted'
    
    # Review Events
    ANNOTATION_REVIEW = 'annotation_review', 'Annotation Reviewed'
    IMAGE_REVIEW = 'image_review', 'Image Reviewed'
    IMAGE_FINALIZE = 'image_finalize', 'Image Finalized'
    
    # Job Events
    JOB_ASSIGN = 'job_assign', 'Job Assigned'
    JOB_COMPLETE = 'job_complete', 'Job Completed'
    
    # AI/Prediction Events
    PREDICTION_GENERATE = 'prediction_generate', 'Predictions Generated'
    PREDICTION_ACCEPT = 'prediction_accept', 'Prediction Accepted'
    PREDICTION_EDIT = 'prediction_edit', 'Prediction Edited'
    PREDICTION_REJECT = 'prediction_reject', 'Prediction Rejected'
    
    # Dataset Events
    DATASET_EXPORT = 'dataset_export', 'Dataset Exported'
    DATASET_VERSION_CREATE = 'dataset_version_create', 'Dataset Version Created'


class ActivityEvent(models.Model):
    """
    Immutable event log for all user activities.
    
    Design Principles:
    - Append-only (never update/delete)
    - Partitioned by organization for multi-tenancy
    - Time-series optimized indexes
    - Minimal foreign keys (use IDs in metadata)
    """
    
    # Primary Key
    event_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True,
        db_index=True,
        help_text="Unique identifier for the event"
    )
    
    # Tenant Isolation (CRITICAL for multi-tenant queries)
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        db_index=True,
        help_text='Organization for tenant isolation'
    )
    
    # Event Classification
    event_type = models.CharField(
        max_length=50,
        choices=ActivityEventType.choices,
        db_index=True
    )
    
    # Actor (who did it)
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        db_index=True,
        help_text='User who performed the action'
    )
    
    # Context (where did it happen)
    project = models.ForeignKey(
        Project,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        db_index=True
    )
    
    # Timestamp (when)
    timestamp = models.DateTimeField(
        auto_now_add=True,
        db_index=True
    )
    
    # Event Details (what exactly happened)
    metadata = models.JSONField(
        default=dict,
        help_text="""
        Flexible JSON field for event-specific data:
        {
            "image_id": "uuid",
            "annotation_id": "uuid",
            "annotation_count": 5,
            "annotation_source": "manual",
            "edit_magnitude": 0.2,
            "confidence": 0.95,
            "job_id": "uuid",
            "status_from": "unannotated",
            "status_to": "annotated",
            "file_size_mb": 2.5,
            "duration_seconds": 120
        }
        """
    )
    
    # Performance Metrics
    duration_ms = models.IntegerField(
        null=True,
        blank=True,
        help_text='Action duration in milliseconds'
    )
    
    # Session Context
    session_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text='Group related actions in same session'
    )
    
    # Source tracking
    source = models.CharField(
        max_length=50,
        default='web',
        help_text='Origin: web, api, mobile, automation'
    )

    class Meta:
        db_table = 'activity_event'
        verbose_name = 'Activity Event'
        verbose_name_plural = 'Activity Events'
        
        # Optimized for time-series queries
        ordering = ['-timestamp']
        
        indexes = [
            # Multi-tenant + time-range queries (MOST IMPORTANT)
            models.Index(fields=['organization', '-timestamp']),
            models.Index(fields=['organization', 'event_type', '-timestamp']),
            models.Index(fields=['organization', 'user', '-timestamp']),
            models.Index(fields=['organization', 'project', '-timestamp']),
            
            # User activity tracking
            models.Index(fields=['user', '-timestamp']),
            models.Index(fields=['user', 'event_type', '-timestamp']),
            
            # Project-specific queries
            models.Index(fields=['project', '-timestamp']),
            
            # Session analysis
            models.Index(fields=['session_id', 'timestamp']),
        ]
        
        # Partition by organization (PostgreSQL 11+)
        # This should be done via migration for production
        # 'partition_by': 'RANGE (organization_id)'

    def __str__(self):
        return f"{self.user} - {self.event_type} @ {self.timestamp}"
    
    def __repr__(self):
        return f"<ActivityEvent: {self.event_type} by {self.user_id}>"


# ============================================================================
# 2. PRE-AGGREGATED METRICS (Fast Reads)
# ============================================================================

class UserActivityMetrics(models.Model):
    """
    Pre-computed user activity metrics updated via background tasks.
    
    Aggregation Strategy:
    - Updated hourly/daily via Celery tasks
    - Atomic increments for real-time counters
    - Separate tables for different time granularities
    """
    
    # Composite Key
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        help_text='Null = organization-wide metrics'
    )
    
    # Time Period
    period_start = models.DateTimeField(db_index=True)
    period_end = models.DateTimeField(db_index=True)
    granularity = models.CharField(
        max_length=10,
        choices=[
            ('hour', 'Hourly'),
            ('day', 'Daily'),
            ('week', 'Weekly'),
            ('month', 'Monthly'),
        ],
        default='day'
    )
    
    # === IMAGE METRICS ===
    images_uploaded = models.IntegerField(default=0)
    images_added_to_project = models.IntegerField(default=0)
    total_upload_size_mb = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0
    )
    
    # === ANNOTATION METRICS ===
    annotations_created = models.IntegerField(default=0)
    annotations_updated = models.IntegerField(default=0)
    annotations_deleted = models.IntegerField(default=0)
    
    # Manual vs AI-assisted
    manual_annotations = models.IntegerField(default=0)
    ai_predictions_accepted = models.IntegerField(default=0)
    ai_predictions_edited = models.IntegerField(default=0)
    ai_predictions_rejected = models.IntegerField(default=0)
    
    # === REVIEW METRICS ===
    images_reviewed = models.IntegerField(default=0)
    images_approved = models.IntegerField(default=0)
    images_rejected = models.IntegerField(default=0)
    images_finalized = models.IntegerField(default=0)
    
    annotations_reviewed = models.IntegerField(default=0)
    
    # === QUALITY METRICS ===
    avg_annotation_time_seconds = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True
    )
    avg_edit_magnitude = models.DecimalField(
        max_digits=5,
        decimal_places=4,
        null=True,
        blank=True,
        help_text='Average change magnitude for edited predictions'
    )
    
    # === JOB METRICS ===
    jobs_completed = models.IntegerField(default=0)
    
    # Timestamps
    last_activity = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'user_activity_metrics'
        verbose_name = 'User Activity Metrics'
        verbose_name_plural = 'User Activity Metrics'
        
        unique_together = [
            ('user', 'organization', 'project', 'period_start', 'granularity')
        ]
        
        indexes = [
            # Organization dashboards
            models.Index(fields=['organization', 'granularity', '-period_start']),
            models.Index(fields=['organization', 'user', '-period_start']),
            
            # Project leaderboards
            models.Index(fields=['project', '-annotations_created']),
            models.Index(fields=['project', '-images_reviewed']),
            
            # User profiles
            models.Index(fields=['user', 'organization', '-period_start']),
        ]
        
        ordering = ['-period_start']

    def __str__(self):
        project_name = self.project.name if self.project else 'All Projects'
        return f"{self.user.username} - {project_name} ({self.period_start.date()})"


class ProjectActivityMetrics(models.Model):
    """
    Project-level aggregate metrics for dashboard KPIs.
    """
    
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    
    # Time Period
    period_start = models.DateTimeField(db_index=True)
    period_end = models.DateTimeField(db_index=True)
    granularity = models.CharField(max_length=10, default='day')
    
    # === OVERALL PROGRESS ===
    total_images = models.IntegerField(default=0)
    images_unannotated = models.IntegerField(default=0)
    images_annotated = models.IntegerField(default=0)
    images_reviewed = models.IntegerField(default=0)
    images_finalized = models.IntegerField(default=0)
    
    total_annotations = models.IntegerField(default=0)
    manual_annotations = models.IntegerField(default=0)
    ai_predictions = models.IntegerField(default=0)
    
    # === QUALITY METRICS ===
    untouched_predictions = models.IntegerField(
        default=0,
        help_text='AI predictions not reviewed/edited'
    )
    edited_predictions = models.IntegerField(default=0)
    rejected_predictions = models.IntegerField(default=0)
    
    avg_annotations_per_image = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True,
        blank=True
    )
    
    # === VELOCITY METRICS ===
    annotations_per_hour = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True
    )
    
    # === CONTRIBUTOR STATS ===
    active_users = models.IntegerField(
        default=0,
        help_text='Users with activity in this period'
    )
    top_contributor_id = models.UUIDField(null=True, blank=True)
    
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'project_activity_metrics'
        unique_together = [('project', 'period_start', 'granularity')]
        indexes = [
            models.Index(fields=['project', '-period_start']),
            models.Index(fields=['organization', '-period_start']),
        ]
        ordering = ['-period_start']

    def __str__(self):
        return f"{self.project.name} - {self.period_start.date()}"


# ============================================================================
# 3. REAL-TIME COUNTERS (Atomic Operations)
# ============================================================================

class UserSessionActivity(models.Model):
    """
    Real-time tracking of current user sessions.
    Short TTL, cleared after 24 hours.
    """
    
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    
    # Session metadata
    started_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    
    # Real-time counters (atomic updates)
    actions_count = models.IntegerField(default=0)
    annotations_created = models.IntegerField(default=0)
    images_processed = models.IntegerField(default=0)
    
    # Session state
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'user_session_activity'
        indexes = [
            models.Index(fields=['user', '-started_at']),
            models.Index(fields=['organization', 'is_active']),
            models.Index(fields=['-last_activity']),  # For cleanup
        ]

    def __str__(self):
        return f"Session {self.session_id} - {self.user.username}"