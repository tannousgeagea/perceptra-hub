"""
API Keys models for programmatic access.
"""
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import timedelta
import secrets

User = get_user_model()


class APIKey(models.Model):
    """
    API Keys for programmatic access.
    Supports org-wide and user-specific keys with scoped permissions.
    """
    SCOPE_CHOICES = [
        ('organization', 'Organization-wide'),
        ('user', 'User-specific'),
    ]
    
    PERMISSION_CHOICES = [
        ('read', 'Read-only'),
        ('write', 'Read & Write'),
        ('admin', 'Admin (Full Access)'),
    ]
    
    # Key fields
    key = models.CharField(max_length=255, unique=True, db_index=True)
    prefix = models.CharField(max_length=16, db_index=True)  # For display: "vsk_live..."
    hashed_key = models.CharField(max_length=128)  # Store hashed version for security
    
    # Metadata
    name = models.CharField(max_length=255, help_text="Friendly name for this key")
    description = models.TextField(blank=True, null=True)
    
    # Ownership
    organization = models.ForeignKey(
        'organizations.Organization',
        on_delete=models.CASCADE,
        related_name='api_keys'
    )
    
    scope = models.CharField(
        max_length=20,
        choices=SCOPE_CHOICES,
        default='organization',
        help_text="Organization-wide or user-specific"
    )
    
    # Null if org-wide, set if user-specific
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='api_keys',
        help_text="If user-specific, which user this key belongs to"
    )
    
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_api_keys'
    )
    
    # Permissions
    permission = models.CharField(
        max_length=20,
        choices=PERMISSION_CHOICES,
        default='read',
        help_text="Permission level: read, write, or admin"
    )
    
    # Status
    is_active = models.BooleanField(default=True)
    
    # Usage tracking
    last_used_at = models.DateTimeField(null=True, blank=True)
    usage_count = models.IntegerField(default=0)
    
    # Expiration
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(
        help_text="Key expiration date (default: 90 days from creation)"
    )
    
    # Rate limiting
    rate_limit_per_minute = models.IntegerField(
        default=60,
        help_text="Max requests per minute (0 = unlimited)"
    )
    rate_limit_per_hour = models.IntegerField(
        default=1000,
        help_text="Max requests per hour (0 = unlimited)"
    )
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['key']),
            models.Index(fields=['prefix']),
            models.Index(fields=['organization', 'is_active']),
            models.Index(fields=['hashed_key']),
        ]
        verbose_name = 'API Key'
        verbose_name_plural = 'API Keys'
    
    def __str__(self):
        return f"{self.name} ({self.prefix}...)"
    
    @classmethod
    def generate_key(cls, prefix="vsk_live"):
        """
        Generate API key in format: vsk_live_<random>
        Returns: (full_key, prefix_for_display, hashed_key)
        """
        import hashlib
        
        random_part = secrets.token_urlsafe(32)  # 43 chars base64
        full_key = f"{prefix}_{random_part}"
        display_prefix = full_key[:12]  # "vsk_live_abc"
        
        # Hash the key for storage (we only store hashed version)
        hashed = hashlib.sha256(full_key.encode()).hexdigest()
        
        return full_key, display_prefix, hashed
    
    @staticmethod
    def hash_key(key: str) -> str:
        """Hash an API key for comparison"""
        import hashlib
        return hashlib.sha256(key.encode()).hexdigest()
    
    def is_valid(self) -> bool:
        """Check if key is valid and not expired"""
        if not self.is_active:
            return False
        
        if self.expires_at and self.expires_at < timezone.now():
            return False
        
        return True
    
    def has_permission(self, required_permission: str) -> bool:
        """
        Check if key has required permission.
        Permission hierarchy: admin > write > read
        """
        permission_levels = {
            'read': 1,
            'write': 2,
            'admin': 3
        }
        
        current_level = permission_levels.get(self.permission, 0)
        required_level = permission_levels.get(required_permission, 0)
        
        return current_level >= required_level
    
    def increment_usage(self):
        """Increment usage count and update last_used_at"""
        from django.db.models import F
        
        self.__class__.objects.filter(pk=self.pk).update(
            usage_count=F('usage_count') + 1,
            last_used_at=timezone.now()
        )
    
    def revoke(self):
        """Revoke (deactivate) this API key"""
        self.is_active = False
        self.save(update_fields=['is_active'])
    
    def renew(self, days=90):
        """Extend expiration by specified days"""
        self.expires_at = timezone.now() + timedelta(days=days)
        self.save(update_fields=['expires_at'])
    
    def save(self, *args, **kwargs):
        # Set default expiration (90 days) if not provided
        if not self.expires_at:
            self.expires_at = timezone.now() + timedelta(days=90)
        
        super().save(*args, **kwargs)


class APIKeyUsageLog(models.Model):
    """
    Track API key usage for rate limiting and analytics.
    """
    api_key = models.ForeignKey(
        APIKey,
        on_delete=models.CASCADE,
        related_name='usage_logs'
    )
    
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    
    # Request details
    endpoint = models.CharField(max_length=255)
    method = models.CharField(max_length=10)  # GET, POST, etc.
    status_code = models.IntegerField()
    response_time_ms = models.IntegerField(null=True)
    
    # IP tracking
    ip_address = models.GenericIPAddressField(null=True)
    user_agent = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['api_key', 'timestamp']),
            models.Index(fields=['timestamp']),
        ]
        verbose_name = 'API Key Usage Log'
        verbose_name_plural = 'API Key Usage Logs'
    
    def __str__(self):
        return f"{self.api_key.prefix}... - {self.method} {self.endpoint} ({self.timestamp})"


class APIKeyRateLimit(models.Model):
    """
    Track rate limiting per API key (time-windowed counters).
    Uses Redis in production, DB for development.
    """
    api_key = models.ForeignKey(
        APIKey,
        on_delete=models.CASCADE,
        related_name='rate_limits'
    )
    
    window_start = models.DateTimeField(db_index=True)
    window_type = models.CharField(
        max_length=10,
        choices=[
            ('minute', 'Per Minute'),
            ('hour', 'Per Hour'),
        ]
    )
    
    request_count = models.IntegerField(default=0)
    
    class Meta:
        unique_together = [['api_key', 'window_start', 'window_type']]
        indexes = [
            models.Index(fields=['api_key', 'window_type', 'window_start']),
        ]
        verbose_name = 'API Key Rate Limit'
        verbose_name_plural = 'API Key Rate Limits'
    
    def __str__(self):
        return f"{self.api_key.prefix}... - {self.window_type} ({self.request_count} requests)"