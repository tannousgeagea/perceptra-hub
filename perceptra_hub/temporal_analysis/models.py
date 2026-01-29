from django.db import models
from django.contrib.auth import get_user_model
from projects.models import Project
from ml_models.models import ModelVersion

User = get_user_model()


class MetricSnapshot(models.Model):
    """Historical snapshots of evaluation metrics"""
    
    # Identity - Link to existing models
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='metric_snapshots',
        db_index=True
    )
    model_version = models.ForeignKey(
        ModelVersion,
        on_delete=models.CASCADE,
        related_name='metric_snapshots',
        null=True,
        blank=True,
        help_text="Specific model version being evaluated"
    )
    snapshot_date = models.DateTimeField(auto_now_add=True, db_index=True)
    
    # Dataset context
    total_images = models.IntegerField(default=0)
    reviewed_images = models.IntegerField(default=0)
    total_annotations = models.IntegerField(default=0)
    
    # Core metrics
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    
    # Counts
    tp = models.IntegerField(default=0)
    fp = models.IntegerField(default=0)
    fn = models.IntegerField(default=0)
    
    # Quality metrics
    edit_rate = models.FloatField(default=0.0)
    hallucination_rate = models.FloatField(default=0.0)
    mean_confidence = models.FloatField(null=True, blank=True)
    mean_localization_iou = models.FloatField(null=True, blank=True)
    
    # Metadata
    computation_time_seconds = models.FloatField(default=0.0)
    
    class Meta:
        db_table = 'metric_snapshot'
        ordering = ['-snapshot_date']
        indexes = [
            models.Index(fields=['project', 'model_version', 'snapshot_date']),
            models.Index(fields=['project', '-snapshot_date']),
        ]
    
    def __str__(self):
        model_str = f" - {self.model_version.display_name}" if self.model_version else ""
        return f"{self.project.name}{model_str} - {self.snapshot_date.date()} (F1: {self.f1_score:.2%})"


class MetricAlert(models.Model):
    """Alerts for metric degradation"""
    
    SEVERITY_CHOICES = [
        ('critical', 'Critical'),
        ('warning', 'Warning'),
        ('info', 'Info'),
    ]
    
    # Context - Link to existing models
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='metric_alerts',
        db_index=True
    )
    model_version = models.ForeignKey(
        ModelVersion,
        on_delete=models.CASCADE,
        related_name='metric_alerts',
        null=True,
        blank=True
    )
    alert_date = models.DateTimeField(auto_now_add=True, db_index=True)
    
    # Alert details
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    metric_name = models.CharField(max_length=50)
    current_value = models.FloatField()
    previous_value = models.FloatField(null=True, blank=True)
    threshold_value = models.FloatField()
    change_percent = models.FloatField(null=True, blank=True)
    
    # Status
    is_acknowledged = models.BooleanField(default=False, db_index=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    acknowledged_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='acknowledged_alerts'
    )
    
    # Message
    message = models.TextField()
    
    class Meta:
        db_table = 'metric_alert'
        ordering = ['-alert_date']
        indexes = [
            models.Index(fields=['project', 'is_acknowledged']),
            models.Index(fields=['severity', '-alert_date']),
        ]
    
    def __str__(self):
        return f"{self.severity.upper()}: {self.metric_name} - {self.project.name}"