"""
Compute profiles for ML training - defines WHERE and HOW to train models.
"""
from django.db import models
from organizations.models import Organization
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _

User = get_user_model()


class ComputeProvider(models.Model):
    """Available compute providers for training"""
    PROVIDER_CHOICES = [
        ('platform-gpu', 'Platform GPU Workers'),  # Your on-premise GPUs
        ('platform-cpu', 'Platform CPU Workers'),  # Fallback CPU
        ('aws-sagemaker', 'AWS SageMaker'),
        ('gcp-vertex', 'GCP Vertex AI'),
        ('azure-ml', 'Azure ML'),
        ('kubernetes', 'Kubernetes Cluster'),
        ('modal', 'Modal Labs'),
        ('runpod', 'RunPod'),
    ]
    
    name = models.CharField(max_length=100, unique=True)
    provider_type = models.CharField(max_length=50, choices=PROVIDER_CHOICES)
    description = models.TextField(blank=True)
    
    # System-level config (API keys, endpoints, etc.)
    # Only accessible by platform admins
    system_config = models.JSONField(
        default=dict,
        help_text=_('Provider credentials and global settings (admin only)')
    )
    
    # Available instance types
    available_instances = models.JSONField(
        default=list,
        help_text=_('List of available instance types with specs')
    )
    
    is_active = models.BooleanField(default=True)
    requires_user_credentials = models.BooleanField(
        default=False,
        help_text=_('Whether users need to provide their own credentials')
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'compute_provider'
        verbose_name_plural = 'Compute Providers'
    
    def __str__(self):
        return f"{self.name} ({self.provider_type})"


class ComputeProfile(models.Model):
    """
    Organization's compute configuration for training.
    Defines which compute resources to use and fallback strategy.
    """
    STRATEGY_CHOICES = [
        ('cheapest', 'Cheapest Available'),
        ('fastest', 'Fastest (Best GPU)'),
        ('preferred', 'Preferred Provider'),
        ('queue', 'Queue on Platform GPUs'),
    ]
    
    profile_id = models.CharField(max_length=255, unique=True)
    name = models.CharField(max_length=255)
    
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='compute_profiles'
    )
    
    # Primary compute provider
    provider = models.ForeignKey(
        ComputeProvider,
        on_delete=models.PROTECT,
        related_name='compute_profiles'
    )
    
    # User-provided credentials (if required by provider)
    # e.g., AWS access keys, GCP service account
    user_credentials = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Encrypted user credentials for cloud providers')
    )
    
    # Training configuration
    default_instance_type = models.CharField(
        max_length=100,
        help_text=_('Default instance type (e.g., ml.p3.2xlarge, n1-highmem-8)')
    )
    
    max_concurrent_jobs = models.PositiveIntegerField(
        default=5,
        help_text=_('Max concurrent training jobs')
    )
    
    strategy = models.CharField(
        max_length=20,
        choices=STRATEGY_CHOICES,
        default='queue'
    )
    
    # Fallback providers (if primary fails)
    fallback_providers = models.ManyToManyField(
        ComputeProvider,
        through='ComputeFallback',
        related_name='fallback_for_profiles',
        blank=True
    )
    
    # Cost controls
    max_cost_per_hour = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text=_('Max cost per hour in USD')
    )
    max_training_hours = models.PositiveIntegerField(
        default=24,
        help_text=_('Max hours per training job')
    )
    
    is_active = models.BooleanField(default=True)
    is_default = models.BooleanField(
        default=False,
        help_text=_('Default profile for this organization')
    )
    
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'compute_profile'
        verbose_name_plural = 'Compute Profiles'
        unique_together = [('organization', 'name')]
        indexes = [
            models.Index(fields=['organization', 'is_default']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.organization.name})"
    
    def save(self, *args, **kwargs):
        # Ensure only one default per organization
        if self.is_default:
            ComputeProfile.objects.filter(
                organization=self.organization,
                is_default=True
            ).update(is_default=False)
        super().save(*args, **kwargs)


class ComputeFallback(models.Model):
    """Fallback strategy for compute profiles"""
    profile = models.ForeignKey(ComputeProfile, on_delete=models.CASCADE)
    provider = models.ForeignKey(ComputeProvider, on_delete=models.CASCADE)
    priority = models.PositiveIntegerField(
        help_text=_('Lower number = higher priority (1, 2, 3...)')
    )
    
    class Meta:
        db_table = 'compute_fallback'
        unique_together = [('profile', 'priority')]
        ordering = ['priority']


class TrainingJob(models.Model):
    """
    Extended training job model with compute provider tracking.
    Links TrainingSession to actual compute resources.
    """
    from training.models import TrainingSession
    
    job_id = models.CharField(max_length=255, unique=True)
    
    training_session = models.OneToOneField(
        TrainingSession,
        on_delete=models.CASCADE,
        related_name='training_job'
    )
    
    compute_profile = models.ForeignKey(
        ComputeProfile,
        on_delete=models.PROTECT,
        related_name='training_jobs'
    )
    
    # Actual provider used (might differ from profile's default due to fallback)
    actual_provider = models.ForeignKey(
        ComputeProvider,
        on_delete=models.PROTECT,
        related_name='executed_jobs'
    )
    
    instance_type = models.CharField(max_length=100)
    
    # Provider-specific job identifiers
    external_job_id = models.CharField(
        max_length=500,
        blank=True,
        help_text=_('Job ID in external system (SageMaker ARN, Vertex AI job name, etc.)')
    )
    
    # Cost tracking
    estimated_cost = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True
    )
    actual_cost = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True
    )
    
    # Resource usage
    gpu_hours = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True
    )
    
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'training_job'
        verbose_name_plural = 'Training Jobs'
    
    def __str__(self):
        return f"Job {self.job_id} on {self.actual_provider.name}"