
from django.db import models
from images.models import Image
from ml_models.models import ModelVersion
from projects.models import Version  # Dataset version

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
