from django.db import models
from projects.models import Project, Version as DatasetVersion
from django.core.validators import FileExtensionValidator
from django.contrib.auth import get_user_model
from organizations.models import Organization
from storage.models import StorageProfile
from django.utils.translation import gettext_lazy as _

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
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='model_tags',
        help_text=_('Organization that owns this tag')
    )
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True, null=True)
    color = models.CharField(
        max_length=7,
        default='#3B82F6',
        help_text=_('Hex color for UI display')
    )
    created_at = models.DateTimeField(auto_now_add=True)


    class Meta:
        db_table = 'model_tag'
        verbose_name_plural = 'Model Tags'
        unique_together = [('organization', 'name')]
        indexes = [
            models.Index(fields=['organization', 'name']),
        ]

    def __str__(self):
        return f"{self.name} ({self.organization.name})"


class Model(models.Model):
    """
    ML Model blueprint - represents a model architecture/configuration.
    Multiple versions can be trained from this blueprint.
    """
    model_id = models.CharField(
        max_length=255,
        unique=True,
        help_text=_('Unique identifier for this model')
    )
    name = models.CharField(
        max_length=255,
        help_text=_('Human-readable model name')
    )
    
    # Relationships
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='ml_models',
        help_text=_('Organization that owns this model')
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='ml_models',
        help_text=_('Project this model belongs to')
    )
    task = models.ForeignKey(
        ModelTask,
        on_delete=models.PROTECT,
        help_text=_('Type of task this model performs')
    )
    framework = models.ForeignKey(
        ModelFramework,
        on_delete=models.PROTECT,
        help_text=_('Framework/architecture used')
    )
    
    # Metadata
    description = models.TextField(blank=True)
    tags = models.ManyToManyField(
        ModelTag,
        blank=True,
        related_name='models'
    )
    
    # Default training configuration
    default_config = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Default training config (batchSize, learningRate, epochs, optimizer, scheduler)')
    )
    
    # Audit fields
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='created_models'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Soft delete
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='deleted_models'
    )
    class Meta:
        db_table = "ml_model"
        verbose_name_plural = "ML Models"
        unique_together = [('organization', 'name')]
        indexes = [
            models.Index(fields=['organization', 'project', 'is_deleted']),
            models.Index(fields=['organization', 'created_at']),
        ]


    def __str__(self):
        return f"{self.name} ({self.task.name}, {self.framework.name})"
    
    def get_latest_version(self):
        """Get the most recent version of this model"""
        return self.versions.filter(is_deleted=False).order_by('-version_number').first()
    
    def get_production_version(self):
        """Get the currently deployed production version"""
        return self.versions.filter(
            is_deleted=False,
            deployment_status='production'
        ).first()

class ModelVersion(models.Model):
    """
    A specific trained instance of a model.
    Each version represents a complete training run with specific data and config.
    """
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('queued', 'Queued'),
        ('training', 'Training'),
        ('trained', 'Trained'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    DEPLOYMENT_STATUS_CHOICES = [
        ('none', 'Not Deployed'),
        ('staging', 'Staging'),
        ('production', 'Production'),
        ('retired', 'Retired'),
    ]
    
    # Primary identifier
    version_id = models.CharField(
        max_length=255,
        unique=True,
        help_text=_('Unique identifier for this version')
    )
    
    # Relationships
    model = models.ForeignKey(
        Model,
        on_delete=models.CASCADE,
        related_name='versions'
    )
    dataset_version = models.ForeignKey(
        DatasetVersion,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name='trained_models',
        help_text=_('Dataset version used for training')
    )
    parent_version = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='derived_versions',
        help_text=_('Base version for transfer learning/fine-tuning')
    )
    
    # Version info
    version_number = models.PositiveIntegerField(
        help_text=_('Sequential version number (1, 2, 3, ...)')
    )
    version_name = models.CharField(
        max_length=255,
        blank=True,
        help_text=_('Optional human-readable version name')
    )
    
    # Storage information (CRITICAL for multi-cloud storage)
    storage_profile = models.ForeignKey(
        StorageProfile,
        on_delete=models.PROTECT,
        related_name='model_versions',
        help_text=_('Storage profile where artifacts are stored')
    )
    
    # Training artifacts (stored as keys, not Django files)
    checkpoint_key = models.CharField(
        max_length=500,
        blank=True,
        help_text=_('Storage key for model checkpoint (e.g., org-123/models/model-1/v1/checkpoint.pt)')
    )
    onnx_model_key = models.CharField(
        max_length=500,
        blank=True,
        help_text=_('Storage key for ONNX model')
    )
    training_logs_key = models.CharField(
        max_length=500,
        blank=True,
        help_text=_('Storage key for training logs file')
    )
    
    # Training configuration and results
    config = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Training configuration (hyperparameters, etc.)')
    )
    metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Training metrics (loss, accuracy, mAP, etc.)')
    )
    
    tags = models.ManyToManyField(
        ModelTag,
        blank=True,
        related_name="model_versions",
        help_text=_("Tags specific to this model version")
    )
    
    # Status tracking
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='draft'
    )
    deployment_status = models.CharField(
        max_length=20,
        choices=DEPLOYMENT_STATUS_CHOICES,
        default='none'
    )
    
    # Error tracking
    error_message = models.TextField(
        blank=True,
        help_text=_('Error message if training failed')
    )
    
    # Deployment metadata
    deployed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text=_('When this version was deployed to production')
    )
    deployed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='deployed_model_versions'
    )
    deployment_config = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Deployment configuration (endpoint, resources, etc.)')
    )
    
    # Audit fields
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='created_model_versions'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Soft delete
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = "model_version"
        verbose_name_plural = "Model Versions"
        unique_together = [('model', 'version_number')]
        indexes = [
            models.Index(fields=['model', 'version_number', 'is_deleted']),
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['deployment_status']),
        ]
        ordering = ['-version_number']

    def __str__(self):
        return f"{self.model.name} v{self.version_number}"
    
    @property
    def display_name(self):
        """Get display name (version_name if set, otherwise v{number})"""
        return self.version_name or f"v{self.version_number}"
    
    def can_deploy(self) -> bool:
        """Check if this version is ready for deployment"""
        return self.status == 'trained' and bool(self.checkpoint_key)
    
    def get_checkpoint_url(self, expiration: int = 3600) -> str:
        """Generate presigned URL for checkpoint download"""
        from storage.services import get_storage_adapter_for_profile
        
        if not self.checkpoint_key:
            return None
        
        if self.storage_profile.backend == "local":
            return f"http://localhost:81/{self.storage_profile.config['base_path']}/{self.checkpoint_key}"
        
        adapter = get_storage_adapter_for_profile(self.storage_profile)
        presigned = adapter.generate_presigned_url(
            self.checkpoint_key,
            expiration=expiration,
            method='GET'
        )
        return presigned.url
    
    def get_logs_url(self, expiration: int = 3600) -> str:
        """Generate presigned URL for logs download"""
        from storage.services import get_storage_adapter_for_profile
        
        if not self.training_logs_key:
            return None
        
        if self.storage_profile.backend == "local":
            return f"http://localhost:81/{self.storage_profile.config['base_path']}/{self.training_logs_key}"
        
        adapter = get_storage_adapter_for_profile(self.storage_profile)
        presigned = adapter.generate_presigned_url(
            self.training_logs_key,
            expiration=expiration,
            method='GET'
        )
        return presigned.url
    
    def mark_as_production(self, user: User): # type: ignore
        """Mark this version as production and demote others"""
        from django.utils import timezone
        
        # Demote other production versions
        ModelVersion.objects.filter(
            model=self.model,
            deployment_status='production'
        ).update(deployment_status='retired')
        
        # Promote this version
        self.deployment_status = 'production'
        self.deployed_at = timezone.now()
        self.deployed_by = user
        self.save()