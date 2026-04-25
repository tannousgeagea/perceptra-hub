
from django.db import models
from django.contrib.auth import get_user_model
from images.models import Image
from ml_models.models import ModelVersion
from projects.models import Version  # Dataset version

User = get_user_model()

class PredictionImageResult(models.Model):
    """
    Links a model version's prediction output to a specific validation image.
    """
    model_version = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name="prediction_results"
    )
    dataset_version = models.ForeignKey(
        Version, on_delete=models.CASCADE, related_name="prediction_results"
    )
    image = models.ForeignKey(
        Image, on_delete=models.CASCADE, related_name="prediction_results"
    )
    inference_time = models.FloatField(null=True, blank=True, help_text="Time taken to run inference (in seconds)")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('model_version', 'dataset_version', 'image')
        db_table = "prediction_image_result"
        verbose_name_plural = "Prediction Image Results"

    def __str__(self):
        return f"{self.model_version} | {self.image.image_name}"


class PredictionOverlay(models.Model):
    """
    Stores structured prediction outputs (e.g., bounding boxes or masks) per image.
    """
    prediction_result = models.ForeignKey(
        PredictionImageResult, on_delete=models.CASCADE, related_name="overlays"
    )
    class_id = models.PositiveIntegerField(null=True, blank=True)
    class_label = models.CharField(max_length=255)
    confidence = models.FloatField(default=0.0)
    bbox = models.JSONField(
        blank=True, null=True,
        help_text="Bounding box format: [x_min, y_min, x_max, y_max] (normalized)"
    )
    mask = models.JSONField(
        blank=True, null=True,
        help_text="Optional segmentation mask (e.g., polygon or RLE)"
    )
    overlay_type = models.CharField(
        max_length=50,
        choices=[('bbox', 'Bounding Box'), ('mask', 'Segmentation Mask')],
        default='bbox'
    )
    matched_gt = models.BooleanField(
        default=False,
        help_text="Whether this prediction matched a ground-truth annotation (for metric eval)"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "prediction_overlay"
        verbose_name_plural = "Prediction Overlays"

    def __str__(self):
        return f"{self.class_label} ({self.confidence:.2f})"


class InferenceDeployment(models.Model):
    """
    Records each time a ModelVersion is deployed to / undeployed from the inference service.
    One active deployment per model version at a time (is_active flag).
    """
    TARGET_CHOICES = [
        ('staging', 'Staging'),
        ('production', 'Production'),
    ]

    model_version = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name='deployments'
    )
    target_env = models.CharField(max_length=20, choices=TARGET_CHOICES, default='staging')
    inference_service_url = models.CharField(max_length=500)
    deployed_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='+'
    )
    deployed_at = models.DateTimeField(auto_now_add=True)
    undeployed_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True, db_index=True)

    class Meta:
        db_table = 'inference_deployment'
        verbose_name_plural = 'Inference Deployments'
        indexes = [
            models.Index(fields=['model_version', 'is_active']),
        ]

    def __str__(self):
        return f"{self.model_version} → {self.target_env} ({'active' if self.is_active else 'retired'})"


class ModelEvaluation(models.Model):
    """
    Champion / challenger evaluation comparing a newly trained version (challenger)
    against the current production version (champion).
    Created automatically after training completes.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    RECOMMENDATION_CHOICES = [
        ('promote', 'Promote Challenger'),
        ('keep_champion', 'Keep Champion'),
        ('inconclusive', 'Inconclusive'),
    ]

    evaluation_id = models.CharField(max_length=255, unique=True)
    challenger = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name='challenger_evaluations'
    )
    champion = models.ForeignKey(
        ModelVersion, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='champion_evaluations'
    )
    dataset_version = models.ForeignKey(
        Version, on_delete=models.PROTECT, related_name='model_evaluations'
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', db_index=True)

    challenger_metrics = models.JSONField(default=dict, blank=True)
    champion_metrics = models.JSONField(default=dict, blank=True)

    primary_metric = models.CharField(
        max_length=100, default='f1_score',
        help_text="Metric key used for champion/challenger comparison"
    )
    improvement_delta = models.FloatField(
        null=True, blank=True,
        help_text="challenger_metric - champion_metric (positive = improvement)"
    )
    auto_promote_threshold = models.FloatField(
        default=0.02,
        help_text="Minimum delta required to auto-promote challenger"
    )
    recommendation = models.CharField(
        max_length=20, choices=RECOMMENDATION_CHOICES, null=True, blank=True
    )
    auto_promoted = models.BooleanField(default=False)

    triggered_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='+'
    )
    error_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'model_evaluation'
        verbose_name_plural = 'Model Evaluations'
        indexes = [
            models.Index(fields=['challenger', 'status']),
            models.Index(fields=['status', 'created_at']),
        ]

    def __str__(self):
        return f"Eval {self.evaluation_id[:8]} challenger={self.challenger} status={self.status}"
