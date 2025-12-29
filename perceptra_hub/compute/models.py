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
        ('on-premise-agent', 'On-Premise Agent'),  # User's own GPUs
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
    
class Agent(models.Model):
    """
    User's on-premise training agent.
    Represents a GPU machine running the training agent software.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),      # Registered but not connected yet
        ('ready', 'Ready'),           # Connected and available
        ('busy', 'Busy'),             # Currently training
        ('offline', 'Offline'),       # Not connected (timeout)
        ('error', 'Error'),           # Error state
    ]
    
    agent_id = models.CharField(
        max_length=255,
        unique=True,
        help_text=_('Unique identifier for this agent')
    )
    
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='agents',
        help_text=_('Organization that owns this agent')
    )
    
    name = models.CharField(
        max_length=255,
        help_text=_('Human-readable name for this agent')
    )
    
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )
    
    # GPU information
    gpu_info = models.JSONField(
        default=list,
        help_text=_('List of GPUs available on this agent')
    )
    
    # System information
    system_info = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('System specs (CPU, RAM, OS, etc.)')
    )
    
    # Connection details
    last_heartbeat = models.DateTimeField(
        null=True,
        blank=True,
        help_text=_('Last time agent sent heartbeat')
    )
    
    ip_address = models.GenericIPAddressField(
        null=True,
        blank=True,
        help_text=_('Agent IP address')
    )
    
    version = models.CharField(
        max_length=50,
        blank=True,
        help_text=_('Agent software version')
    )
    
    # Capabilities
    max_concurrent_jobs = models.PositiveIntegerField(
        default=1,
        help_text=_('Maximum concurrent training jobs')
    )
    
    # Metadata
    notes = models.TextField(
        blank=True,
        help_text=_('Admin notes about this agent')
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='created_agents'
    )
    
    class Meta:
        db_table = 'agent'
        verbose_name_plural = 'Agents'
        indexes = [
            models.Index(fields=['organization', 'status']),
            models.Index(fields=['last_heartbeat']),
        ]
        unique_together = [('organization', 'name')]
    
    def __str__(self):
        return f"{self.name} ({self.organization.name})"
    
    @property
    def is_online(self) -> bool:
        """Check if agent is online (heartbeat within last 2 minutes)"""
        from django.utils import timezone
        from datetime import timedelta
        
        if not self.last_heartbeat:
            return False
        
        return self.last_heartbeat > timezone.now() - timedelta(minutes=2)
    
    @property
    def gpu_count(self) -> int:
        """Get number of GPUs"""
        return len(self.gpu_info) if isinstance(self.gpu_info, list) else 0
    
    def update_heartbeat(self):
        """Update last heartbeat timestamp"""
        from django.utils import timezone
        
        self.last_heartbeat = timezone.now()
        if self.status == 'offline':
            self.status = 'ready'
        self.save(update_fields=['last_heartbeat', 'status', 'updated_at'])


class AgentAPIKey(models.Model):
    """
    API keys for agent authentication.
    Each agent needs a key to connect to the platform.
    """
    key_id = models.CharField(
        max_length=255,
        unique=True,
        help_text=_('Public key identifier (e.g., org_123_agent_abc)')
    )
    
    key_hash = models.CharField(
        max_length=255,
        help_text=_('Hashed secret key')
    )
    
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='agent_api_keys'
    )
    
    agent = models.ForeignKey(
        Agent,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='api_keys',
        help_text=_('Agent this key is for (null if not yet assigned)')
    )
    
    name = models.CharField(
        max_length=255,
        help_text=_('Human-readable name for this key')
    )
    
    is_active = models.BooleanField(default=True)
    
    last_used = models.DateTimeField(
        null=True,
        blank=True,
        help_text=_('Last time this key was used')
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    
    expires_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text=_('Optional expiration date')
    )
    
    class Meta:
        db_table = 'agent_api_key'
        verbose_name_plural = 'Agent API Keys'
        indexes = [
            models.Index(fields=['organization', 'is_active']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.key_id})"
    
    @staticmethod
    def generate_key() -> tuple[str, str]:
        """
        Generate a new API key.
        Returns (key_id, secret_key)
        """
        import secrets
        import hashlib
        
        # Generate secret (32 bytes = 256 bits)
        secret = secrets.token_urlsafe(32)
        
        # Generate key_id
        key_id = f"agent_{secrets.token_hex(8)}"
        
        # Hash secret for storage
        key_hash = hashlib.sha256(secret.encode()).hexdigest()
        
        return key_id, secret, key_hash
    
    def verify_secret(self, secret: str) -> bool:
        """Verify if provided secret matches stored hash"""
        import hashlib
        
        provided_hash = hashlib.sha256(secret.encode()).hexdigest()
        return provided_hash == self.key_hash
    
    @property
    def is_expired(self) -> bool:
        """Check if key is expired"""
        from django.utils import timezone
        
        if not self.expires_at:
            return False
        return timezone.now() > self.expires_at