from django.db import models
from django.contrib.auth import get_user_model
from django.core.validators import MinLengthValidator, RegexValidator
from django.utils.translation import gettext_lazy as _
from django.utils.text import slugify
from django.core.exceptions import ValidationError
import uuid

User = get_user_model()


class SubscriptionPlan(models.TextChoices):
    """Subscription plan tiers for organizations."""
    FREE = 'free', _('Free')
    STARTER = 'starter', _('Starter')
    PROFESSIONAL = 'professional', _('Professional')
    ENTERPRISE = 'enterprise', _('Enterprise')


class OrganizationStatus(models.TextChoices):
    """Organization account status."""
    ACTIVE = 'active', _('Active')
    SUSPENDED = 'suspended', _('Suspended')
    TRIAL = 'trial', _('Trial')
    INACTIVE = 'inactive', _('Inactive')


class Organization(models.Model):
    """
    Organization model for multi-tenant architecture.
    
    Each organization represents a separate tenant with isolated data,
    users, projects, and resources.
    """
    
    # Unique identifiers
    org_id = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        editable=False,
        db_index=True,
        help_text=_('Unique UUID identifier for the organization')
    )
    
    # Basic information
    name = models.CharField(
        max_length=255,
        unique=True,
        validators=[MinLengthValidator(2)],
        help_text=_('Organization name')
    )
    
    slug = models.SlugField(
        max_length=255,
        unique=True,
        validators=[
            RegexValidator(
                regex=r'^[a-z0-9-]+$',
                message=_('Slug must contain only lowercase letters, numbers, and hyphens')
            )
        ],
        help_text=_('URL-friendly identifier')
    )
    
    display_name = models.CharField(
        max_length=255,
        blank=True,
        help_text=_('Public-facing display name (defaults to name)')
    )
    
    description = models.TextField(
        blank=True,
        null=True,
        help_text=_('Organization description')
    )
    
    # Contact information
    email = models.EmailField(
        blank=True,
        null=True,
        help_text=_('Organization contact email')
    )
    
    website = models.URLField(
        blank=True,
        null=True,
        help_text=_('Organization website')
    )
    
    # Branding
    logo_url = models.URLField(
        blank=True,
        null=True,
        max_length=500,
        help_text=_('Organization logo URL')
    )
    
    # Ownership and management
    owner = models.ForeignKey(
        User,
        on_delete=models.PROTECT,
        related_name='owned_organizations',
        help_text=_('Organization owner')
    )
    
    # Subscription and billing
    subscription_plan = models.CharField(
        max_length=20,
        choices=SubscriptionPlan.choices,
        default=SubscriptionPlan.FREE,
        db_index=True,
        help_text=_('Current subscription plan')
    )
    
    subscription_start_date = models.DateTimeField(
        null=True,
        blank=True,
        help_text=_('When the current subscription started')
    )
    
    subscription_end_date = models.DateTimeField(
        null=True,
        blank=True,
        help_text=_('When the current subscription ends')
    )
    
    trial_end_date = models.DateTimeField(
        null=True,
        blank=True,
        help_text=_('Trial period end date')
    )
    
    # Status
    status = models.CharField(
        max_length=20,
        choices=OrganizationStatus.choices,
        default=OrganizationStatus.ACTIVE,
        db_index=True,
        help_text=_('Organization account status')
    )
    
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        help_text=_('Whether the organization is active')
    )
    
    # Resource limits (based on subscription plan)
    max_projects = models.IntegerField(
        default=5,
        help_text=_('Maximum number of projects allowed')
    )
    
    max_storage_gb = models.IntegerField(
        default=10,
        help_text=_('Maximum storage in GB')
    )
    
    max_users = models.IntegerField(
        default=5,
        help_text=_('Maximum number of users')
    )
    
    max_models = models.IntegerField(
        default=10,
        help_text=_('Maximum number of trained models')
    )
    
    # Usage tracking
    current_storage_gb = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=0,
        help_text=_('Current storage usage in GB')
    )
    
    # Settings and configuration
    settings = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Organization-specific settings and configurations')
    )
    
    # Metadata
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Additional metadata')
    )
    
    # Timestamps and audit
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='organizations_created'
    )
    
    updated_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='organizations_updated'
    )

    class Meta:
        db_table = "organization"
        verbose_name = "Organization"
        verbose_name_plural = "Organizations"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['slug', 'is_active']),
            models.Index(fields=['status', 'is_active']),
            models.Index(fields=['subscription_plan', 'status']),
            models.Index(fields=['owner', 'is_active']),
        ]

    def __str__(self):
        return self.display_name or self.name
    
    def __repr__(self):
        return f"<Organization: {self.slug} ({self.name})>"
    
    def save(self, *args, **kwargs):
        """Override save to auto-generate slug and set defaults."""
        # Auto-generate slug if not provided
        if not self.slug:
            self.slug = slugify(self.name)
            
            # Ensure unique slug
            original_slug = self.slug
            counter = 1
            while Organization.objects.filter(slug=self.slug).exists():
                self.slug = f"{original_slug}-{counter}"
                counter += 1
        
        # Set display_name to name if not provided
        if not self.display_name:
            self.display_name = self.name
        
        # Validate before saving
        self.full_clean()
        super().save(*args, **kwargs)
    
    def clean(self):
        """Validate model data."""
        super().clean()
        
        # Validate subscription dates
        if self.subscription_start_date and self.subscription_end_date:
            if self.subscription_end_date <= self.subscription_start_date:
                raise ValidationError({
                    'subscription_end_date': _('End date must be after start date.')
                })
        
        # Validate resource limits are positive
        if self.max_projects < 0:
            raise ValidationError({
                'max_projects': _('Maximum projects cannot be negative.')
            })
        
        if self.max_storage_gb < 0:
            raise ValidationError({
                'max_storage_gb': _('Maximum storage cannot be negative.')
            })
        
        if self.max_users < 1:
            raise ValidationError({
                'max_users': _('Must allow at least one user.')
            })
    
    # Utility methods
    @property
    def is_trial(self):
        """Check if organization is in trial period."""
        from django.utils import timezone
        return (
            self.status == OrganizationStatus.TRIAL and
            self.trial_end_date and
            self.trial_end_date > timezone.now()
        )
    
    @property
    def is_subscription_active(self):
        """Check if subscription is currently active."""
        from django.utils import timezone
        if not self.subscription_end_date:
            return True  # No end date means perpetual
        return self.subscription_end_date > timezone.now()
    
    @property
    def storage_usage_percentage(self):
        """Calculate storage usage as percentage."""
        if self.max_storage_gb == 0:
            return 0
        return (float(self.current_storage_gb) / self.max_storage_gb) * 100
    
    def get_project_count(self):
        """Get number of active projects."""
        return self.projects.filter(is_active=True, is_deleted=False).count()
    
    def get_user_count(self):
        """Get number of members."""
        return self.members.filter(is_active=True).count()
    
    def get_model_count(self):
        """Get number of trained models."""
        # Assuming you have a Model model related to projects
        return sum(
            project.models.count() 
            for project in self.projects.filter(is_active=True, is_deleted=False)
        )
    
    def can_create_project(self):
        """Check if organization can create more projects."""
        return self.get_project_count() < self.max_projects
    
    def can_add_user(self):
        """Check if organization can add more users."""
        return self.get_user_count() < self.max_users
    
    def has_storage_capacity(self, size_gb):
        """Check if organization has enough storage capacity."""
        return (self.current_storage_gb + size_gb) <= self.max_storage_gb
    
    def update_storage_usage(self, delta_gb):
        """Update storage usage."""
        self.current_storage_gb += delta_gb
        if self.current_storage_gb < 0:
            self.current_storage_gb = 0
        self.save(update_fields=['current_storage_gb', 'updated_at'])
    
    def suspend(self, reason=None):
        """Suspend the organization."""
        self.status = OrganizationStatus.SUSPENDED
        self.is_active = False
        if reason and self.metadata:
            self.metadata['suspension_reason'] = reason
        self.save(update_fields=['status', 'is_active', 'metadata', 'updated_at'])
    
    def activate(self):
        """Activate the organization."""
        self.status = OrganizationStatus.ACTIVE
        self.is_active = True
        if self.metadata and 'suspension_reason' in self.metadata:
            del self.metadata['suspension_reason']
        self.save(update_fields=['status', 'is_active', 'metadata', 'updated_at'])
    
    def upgrade_plan(self, new_plan, max_projects=None, max_storage_gb=None, 
                     max_users=None, max_models=None):
        """
        Upgrade organization to a new subscription plan.
        
        Args:
            new_plan: New SubscriptionPlan value
            max_projects: New project limit (optional)
            max_storage_gb: New storage limit (optional)
            max_users: New user limit (optional)
            max_models: New model limit (optional)
        """
        from django.utils import timezone
        
        self.subscription_plan = new_plan
        self.subscription_start_date = timezone.now()
        
        if max_projects is not None:
            self.max_projects = max_projects
        if max_storage_gb is not None:
            self.max_storage_gb = max_storage_gb
        if max_users is not None:
            self.max_users = max_users
        if max_models is not None:
            self.max_models = max_models
        
        self.save()