from django.db import models
from projects.models import Version, VersionImage


class Augmentation(models.Model):
    """Defines different types of augmentations (e.g., Flip, Rotation, Blur)"""
    name = models.CharField(max_length=100, unique=True)
    title = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    thumbnail = models.ImageField(upload_to='augmentations/', null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'augmentation'
        verbose_name_plural = "Augmentations"

    def __str__(self):
        return self.name


class AugmentationParameter(models.Model):
    """Defines all possible parameters that can be used in different augmentations"""
    
    PARAMETER_TYPE_CHOICES = [
        ('float', 'Float'),
        ('int', 'Integer'),
        ('bool', 'Boolean'),
        ('choice', 'Choice'),
    ]

    name = models.CharField(max_length=100, unique=True)
    parameter_type = models.CharField(max_length=10, choices=PARAMETER_TYPE_CHOICES)
    choices = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'augmentation_parameter'
        verbose_name_plural = "Augmentation Parameters"

    def __str__(self):
        return self.name


class AugmentationParameterAssignment(models.Model):
    """Assigns specific parameters to an augmentation with default values"""
    augmentation = models.ForeignKey(Augmentation, on_delete=models.CASCADE, related_name="assigned_parameters")
    parameter = models.ForeignKey(AugmentationParameter, on_delete=models.CASCADE, related_name="assigned_augmentations")
    
    default_value = models.CharField(max_length=50)
    min_value = models.FloatField(null=True, blank=True)
    max_value = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'augmentation_parameter_assignment'
        verbose_name_plural = "Augmentation Parameter Assignments"

    def __str__(self):
        return f"{self.augmentation.name} - {self.parameter.name}"


class DatasetAugmentation(models.Model):
    """Tracks which augmentations are applied to a dataset version"""
    version = models.OneToOneField(Version, on_delete=models.CASCADE, related_name="dataset_augmentation")
    augmentations = models.ManyToManyField(Augmentation, related_name="datasets")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "dataset_augmentation"
        verbose_name_plural = "Dataset Augmentations"

    def __str__(self):
        return f"{self.version.project.name} - v{self.version.version_number}"


class DatasetAugmentationParameter(models.Model):
    """Stores user-defined values for parameters when applying augmentations to a dataset"""
    dataset_augmentation = models.ForeignKey(DatasetAugmentation, on_delete=models.CASCADE, related_name="custom_parameters")
    augmentation = models.ForeignKey(Augmentation, on_delete=models.CASCADE, related_name="dataset_parameters")
    parameter = models.ForeignKey(AugmentationParameter, on_delete=models.CASCADE, related_name="dataset_values")
    value = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'dataset_augmentation_parameter'
        verbose_name_plural = "Dataset Augmentation Parameters"

    def __str__(self):
        return f"{self.dataset_augmentation.version.project.name} - {self.parameter.name}: {self.value}"


class VersionImageAugmentation(models.Model):
    """
    Stores the augmentation result for a single VersionImage.
    Each instance represents one augmented output derived from the original image.
    """
    version_image = models.ForeignKey(
        VersionImage, 
        on_delete=models.CASCADE, 
        related_name='augmentations'
    )
    augmentation_name = models.CharField(max_length=255)
    parameters = models.JSONField(null=True, blank=True)
    augmented_image_file = models.ImageField(upload_to='versions/augmentations/', null=True, blank=True, max_length=255)
    augmented_annotation = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'version_image_augmentation'
        verbose_name_plural = "Version Image Augmentations"

    def __str__(self):
        return f"Augmentation '{self.augmentation_name}' for {self.version_image}"