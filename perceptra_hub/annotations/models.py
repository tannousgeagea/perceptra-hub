
import uuid
from django.db import models
from projects.models import (
    Project,
    ProjectImage
)
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils.translation import gettext_lazy as _
from django.contrib.auth import get_user_model
User = get_user_model()

class AnnotationGroup(models.Model):
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, related_name='annotation_groups'
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'annotation_group'
        verbose_name_plural = "Annotation Groups"

    def __str__(self):
        return f"{self.name} - {self.project.name}"


class AnnotationClass(models.Model):
    annotation_group = models.ForeignKey(
        AnnotationGroup, on_delete=models.CASCADE, related_name='classes'
    )
    class_id = models.PositiveIntegerField()
    name = models.CharField(max_length=255) 
    color = models.CharField(max_length=7, null=True, blank=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'annotation_class'
        verbose_name_plural = 'Annotation Classes'
        unique_together = ('annotation_group', 'class_id')

    def __str__(self):
        return f"{self.class_id} - {self.name} ({self.annotation_group.project})"


# Create your models here.
class AnnotationType(models.Model):
    name = models.CharField(max_length=50, unique=True)  # e.g., "Bounding Box", "Polygon"
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'annotation_type'
        verbose_name_plural = 'Annotation Types'

    def __str__(self):
        return self.name
    
class Annotation(models.Model):
    MINOR_EDIT_THRESHOLD = 0.1
    MAJOR_EDIT_THRESHOLD = 0.4
    SIGNIFICANT_CHANGE_THRESHOLD = 0.4
    ANNOTATION_SOURCE_CHOICES = [
        ('manual', 'Manual Annotation'),
        ('prediction', 'Model Prediction'),
    ]
    
    project_image = models.ForeignKey(
        ProjectImage, on_delete=models.CASCADE, related_name='annotations'
    )
    annotation_type = models.ForeignKey(
        AnnotationType, on_delete=models.SET_NULL, null=True, related_name='annotations'
    )
    annotation_class = models.ForeignKey(
        AnnotationClass, on_delete=models.CASCADE, related_name='annotations'
    )
    data = models.JSONField()
    # area = models.FloatField(editable=False, null=True)
    annotation_uid = models.CharField(max_length=100, unique=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True,related_name='annotations_created')
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, related_name='annotations_updated')
    
    reviewed = models.BooleanField(default=False)
    reviewed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text=_('When annotations were reviewed')
    )
    reviewed_by = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, related_name='annotations_reviewed')
    is_active = models.BooleanField(default=True, db_index=True)
    version = models.IntegerField(default=1)
    
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
        related_name="annotations_deleted"
    )
    

    # Metadata
    annotation_source = models.CharField(max_length=20, choices=ANNOTATION_SOURCE_CHOICES, default='manual', db_index=True)
    confidence = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(1)])

    edit_magnitude = models.FloatField(
        null=True,
        blank=True,
        help_text="Magnitude of edit (0=no change, 1=total change)"
    )
    
    class EditType(models.TextChoices):
        NONE = 'none', 'No Edit'
        MINOR = 'minor', 'Minor Adjustment'
        MAJOR = 'major', 'Major Change'
        CLASS_CHANGE = 'class_change', 'Class Changed'
        DELETED = 'deleted', 'Deleted'
    
    edit_type = models.CharField(
        max_length=20,
        choices=EditType.choices,
        default=EditType.NONE,
        null=True,
        blank=True,
        help_text="Type of edit made"
    )
    
    original_data = models.JSONField(
        null=True,
        blank=True,
        help_text="Original bbox before any edits [x1,y1,x2,y2]"
    )

    original_class = models.ForeignKey(
        AnnotationClass,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='original_predictions'
    )

    model_version = models.CharField(
        max_length=50,
        null=True,
        blank=True,
        db_index=True,
        help_text="e.g. 'yolov8-large-v2.1'"
    )

    class Meta:
        db_table = 'annotation'
        verbose_name_plural = 'Annotations'
        indexes = [
            models.Index(fields=['project_image', 'is_active']),
            models.Index(fields=['project_image', 'is_deleted']),
            
            models.Index(fields=['annotation_source', 'is_active']),
            models.Index(fields=['annotation_source', 'is_deleted']),
            
            models.Index(fields=['annotation_class', 'is_active']),
            models.Index(fields=['annotation_class', 'is_deleted']),
            models.Index(fields=['created_at']),
            models.Index(fields=['annotation_uid']),  # Already unique, but explicit
        ]

    def __str__(self):
        return f"{self.project_image.project.name} - {self.annotation_class.name}"
    
    def clean(self):
        if not (
            isinstance(self.data, list)
            and len(self.data) == 4
        ):
            raise ValidationError("Data must be [xmin, ymin, xmax, ymax]")

        try:
            self.data = [max(0, min(1, float(c))) for c in self.data]
        except (TypeError, ValueError):
            raise ValidationError("Bounding box coordinates must be numeric")    
    
    def _apply_prediction_edit_tracking(self, old: "Annotation"):
        """
        Compare previous and current state and update
        version, edit_type, edit_magnitude, original_*.
        """
        # Store original once
        if not self.original_data:
            if old.data != self.data or old.annotation_class_id != self.annotation_class_id:
                self.original_data = old.data
                self.original_class = old.annotation_class

        class_changed = old.annotation_class_id != self.annotation_class_id
        bbox_change = self.calculate_bbox_change(old.data, self.data)
        self.edit_magnitude = bbox_change

        if self.is_deleted:
            self.edit_type = self.EditType.DELETED
            self.version += 1
            return

        if class_changed:
            self.edit_type = self.EditType.CLASS_CHANGE
            self.version += 1
            return

        if bbox_change >= self.MAJOR_EDIT_THRESHOLD:
            self.edit_type = self.EditType.MAJOR
            self.version += 1
            return

        if bbox_change >= self.MINOR_EDIT_THRESHOLD:
            self.edit_type = self.EditType.MINOR
            return

        self.edit_type = self.EditType.NONE    
    
    def save(self, *args, **kwargs):
        is_update = self.pk is not None
        old = None

        if is_update and self.annotation_source == 'prediction':
            old = Annotation.objects.only(
                "data",
                "annotation_class_id",
                "is_deleted",
                "version",
            ).get(pk=self.pk)

        super().save(*args, **kwargs)
        
        if old:
            self._apply_prediction_edit_tracking(old)
            super().save(
                update_fields=[
                    "version",
                    "edit_type",
                    "edit_magnitude",
                    "original_data",
                    "original_class",
                ]
            )
            
    def soft_delete(self, user=None):
        """Soft delete the project"""
        from django.utils import timezone
        self.is_deleted = True
        self.is_active = False
        self.deleted_at = timezone.now()
        self.deleted_by = user
        self.save(update_fields=['is_deleted', 'is_active', 'deleted_at', 'deleted_by', 'updated_at'])
    

    def calculate_bbox_change(self, old_data, new_data):
        """Calculate how much bbox changed (0-1 scale)."""
        if not old_data or not new_data:
            return 0
        
        # Calculate IoU between old and new
        x1 = max(old_data[0], new_data[0])
        y1 = max(old_data[1], new_data[1])
        x2 = min(old_data[2], new_data[2])
        y2 = min(old_data[3], new_data[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (old_data[2] - old_data[0]) * (old_data[3] - old_data[1])
        area2 = (new_data[2] - new_data[0]) * (new_data[3] - new_data[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return 1 - iou  # Change magnitude (0 = no change, 1 = total change)

class AnnotationAudit(models.Model):
    annotation = models.OneToOneField(
        Annotation,
        on_delete=models.CASCADE,
        related_name="audit"
    )
    evaluation_status = models.CharField(
        max_length=2,
        choices=[
            ('TP', 'True Positive'),
            ('FP', 'False Positive'),
            ('FN', 'False Negative'),
        ],
        null=True,
        blank=True
    )
    was_edited = models.BooleanField(default=False)
    edit_magnitude = models.FloatField(
        null=True,
        blank=True,
        help_text="How much bbox changed (0-1)"
    )
    edit_type = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        help_text="Type of edit: none, minor, major, class_change, deleted"
    )
    
    matched_manual_annotation = models.ForeignKey(
        "Annotation",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="matched_predictions"
    )
    localization_iou = models.FloatField(
        null=True, 
        blank=True,
        help_text="IoU between original_data and current data (for edited TPs)"
    )
    reviewed_at = models.DateTimeField(auto_now=True)
    reviewed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='audits_reviewed'
    )
    
    original_confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="Model confidence before any changes"
    )
    
    class Meta:
        db_table = 'annotation_audit'
        verbose_name_plural = "Annotation Audits"
        indexes = [
            models.Index(fields=['evaluation_status']),
            models.Index(fields=['edit_type']),
        ]

    def __str__(self):
        return f"{self.annotation.id} - {self.evaluation_status or 'Unreviewed'}"



#############################################################
# AI Suggesstions
#############################################################
class SuggestionSession(models.Model):
    """Tracks a batch of AI suggestions for audit/analytics."""
    
    class SourceType(models.TextChoices):
        SAM_AUTO = 'sam_auto', 'SAM Auto-Segment'
        SAM_POINT = 'sam_point', 'SAM Point Prompt'
        SIMILAR_OBJECT = 'similar', 'Similar Object'
        PREVIOUS_FRAME = 'prev_frame', 'Previous Frame'
        LABEL_SUGGEST = 'label', 'Label Suggestion'
    
    suggestion_id = models.UUIDField(default=uuid.uuid4)
    project_image = models.ForeignKey(
        ProjectImage, on_delete=models.CASCADE, related_name='suggestion_sessions'
    )
    source_type = models.CharField(max_length=20, choices=SourceType.choices)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Metrics
    suggestions_generated = models.IntegerField(default=0)
    suggestions_accepted = models.IntegerField(default=0)
    suggestions_rejected = models.IntegerField(default=0)
    
    # Reference data
    source_annotation_uid = models.CharField(max_length=100, null=True, blank=True)
    source_image_id = models.IntegerField(null=True, blank=True)  # for prev_frame
    model_version = models.CharField(max_length=50, null=True, blank=True)
    meta_info = models.JSONField(null=True, blank=True, default=dict)
    
    # Model configuration (stored once per session)
    model_name = models.CharField(
        max_length=20,
        default='sam_v2',
        help_text='SAM model version (sam_v1, sam_v2, sam_v3)'
    )
    model_device = models.CharField(
        max_length=10,
        default='cuda',
        help_text='Device (cuda or cpu)'
    )
    model_precision = models.CharField(
        max_length=10,
        default='fp16',
        help_text='Precision (fp16 or fp32)'
    )
    
    class Meta:
        db_table = 'suggestion_session'
        indexes = [
            models.Index(fields=['project_image', 'created_at']),
            models.Index(fields=['source_type', 'created_at']),
        ]