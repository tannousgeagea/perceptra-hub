from django.db import models
from projects.models import Project, Version as DatasetVersion
from django.core.validators import FileExtensionValidator
from django.contrib.auth import get_user_model

User = get_user_model()

def get_model_artifact_path(instance, filename):
    return f"models/{instance.model.name}/v{instance.version}/artifacts/{filename}"

def get_model_path(instance, filename):
    return f"models/{instance.model.name}/v{instance.version}/{filename}"

class ModelTask(models.Model):
    name = models.CharField(max_length=255)
    description = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "model_task"
        verbose_name_plural = "Model Tasks"
        
    def __str__(self):
        return f"Task: {self.name}"
    
class ModelFramework(models.Model):
    name = models.CharField(max_length=255)
    description = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "model_framework"
        verbose_name_plural = "Model Frameworks"
        
    def __str__(self):
        return f"Framework: {self.name}"

class ModelTag(models.Model):
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True, null=True)

    class Meta:
        db_table = 'model_tag'
        verbose_name_plural = 'Model Tags'

    def __str__(self):
        return self.name

class Model(models.Model):
    name = models.CharField(max_length=255, unique=True)
    task = models.ForeignKey(ModelTask, on_delete=models.RESTRICT)
    framework = models.ForeignKey(ModelFramework, on_delete=models.RESTRICT)
    description = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='models')
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    tags = models.ManyToManyField(ModelTag, blank=True, related_name='models')
    
    class Meta:
        db_table = "ml_model"
        verbose_name_plural = "ML Models"

    def __str__(self):
        return f"{self.name} (Task: {self.task}, Framework: {self.framework})"


class ModelVersion(models.Model):
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('training', 'Training'),
        ('trained', 'Trained'),
        ('failed', 'Failed'),
        ('deployed', 'Deployed'),
    ]
    model = models.ForeignKey(Model, on_delete=models.RESTRICT, related_name='versions')
    version = models.CharField(max_length=50)
    checkpoint = models.FileField(
        upload_to=get_model_artifact_path,
        validators=[FileExtensionValidator(allowed_extensions=['pt', 'pth', 'h5', 'onnx'])],
        max_length=1024, 
        null=True,
        blank=True,
    )
    dataset_version = models.ForeignKey(DatasetVersion, on_delete=models.SET_NULL, null=True, blank=True, related_name='trained_models')
    config = models.JSONField(null=True, blank=True)
    metrics = models.JSONField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    logs = models.FileField(upload_to=get_model_artifact_path, null=True, blank=True)

    class Meta:
        db_table = "model_verison"
        verbose_name_plural = "Model Versions"
        unique_together = ('model', 'version')

    def __str__(self):
        return f"{self.model.name} (Version: {self.version})"