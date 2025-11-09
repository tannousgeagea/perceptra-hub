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

    # Status
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

    class Meta:
        db_table = 'annotation'
        verbose_name_plural = 'Annotations'
        indexes = [
            models.Index(fields=['project_image', 'is_active']),
            models.Index(fields=['annotation_source', 'is_active']),
            models.Index(fields=['annotation_class', 'is_active']),
            models.Index(fields=['created_at']),
            models.Index(fields=['annotation_uid']),  # Already unique, but explicit
        ]

    def __str__(self):
        return f"{self.project_image.project.name} - {self.annotation_class.name}"
    
    def clean(self):
        # Validate data format
        if self.data and isinstance(self.data, list) and len(self.data) == 4:
            self.data = [max(0, min(1, float(coord))) for coord in self.data]
            # Calculate area
            self.area = (self.data[2] - self.data[0]) * (self.data[3] - self.data[1])
        else:
            raise ValidationError("Data must be [xmin, ymin, xmax, ymax]")
    
            # Track edits to predictions
        if self.pk and self.annotation_source == 'prediction':
            try:
                old = Annotation.objects.get(pk=self.pk)
                
                # Check class change
                class_changed = old.annotation_class_id != self.annotation_class_id
                
                # Check bbox change magnitude
                bbox_change = self.calculate_bbox_change(old.data, self.data)
                
                # Significant change threshold (IoU < 0.8 means 20%+ change)
                self.edit_magnitude = bbox_change  # Store for audit
                                
                if class_changed or bbox_change > self.SIGNIFICANT_CHANGE_THRESHOLD or not self.is_active:
                    self.version += 1  # Major edit
                    if class_changed:
                        self.edit_type = self.EditType.CLASS_CHANGE
                    elif not self.is_active:
                        self.edit_type = self.EditType.DELETED
                    elif bbox_change > self.SIGNIFICANT_CHANGE_THRESHOLD:
                        self.edit_type = self.EditType.MAJOR
                else:
                    self.edit_type = self.EditType.MINOR
         
            except Annotation.DoesNotExist:
                pass
    
    
    def save(self, *args, **kwargs):
        self.full_clean()  # Always validate
        super().save(*args, **kwargs)
        
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
    iou = models.FloatField(null=True, blank=True)
    reviewed_at = models.DateTimeField(auto_now=True)
    reviewed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='audits_reviewed'
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
