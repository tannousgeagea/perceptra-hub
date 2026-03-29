"""
API Keys models for programmatic access.
"""
import hmac
import hashlib
import ipaddress
import secrets
import uuid

from django.conf import settings
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import timedelta

User = get_user_model()


def _get_hmac_secret() -> bytes:
    return getattr(settings, 'API_KEY_HMAC_SECRET', settings.SECRET_KEY).encode()


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

    PERMISSION_LEVELS = {
        'read': 1,
        'write': 2,
        'admin': 3,
    }

    # Public-facing UUID (never expose internal pk)
    api_key_id = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        editable=False,
        db_index=True,
    )

    # Key fields — plaintext is NEVER persisted; only prefix + HMAC hash
    key_prefix = models.CharField(max_length=16, db_index=True)
    hashed_key = models.CharField(max_length=128, db_index=True)

    # Metadata
    name = models.CharField(max_length=255, help_text="Friendly name for this key")
    description = models.TextField(blank=True, null=True)

    # Ownership
    organization = models.ForeignKey(
        'organizations.Organization',
        on_delete=models.CASCADE,
        related_name='api_keys',
    )

    scope = models.CharField(
        max_length=20,
        choices=SCOPE_CHOICES,
        default='organization',
        help_text="Organization-wide or user-specific",
    )

    # Scopes (specific API endpoints/resources the key can access)
    scopes = models.JSONField(
        default=list,
        blank=True,
        help_text='List of allowed scopes (e.g. ["projects:read", "images:*"]). Empty = all.',
    )

    # The user this key acts on behalf of. Null = acts as created_by.
    owned_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='api_keys',
        help_text="User this key acts on behalf of (null = acts as creator)",
    )

    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_api_keys',
    )

    # Permissions
    permissions = models.CharField(
        max_length=20,
        choices=PERMISSION_CHOICES,
        default='read',
        help_text="Permission level: read, write, or admin",
    )

    # Status
    is_active = models.BooleanField(default=True)

    # Usage tracking
    last_used_at = models.DateTimeField(null=True, blank=True)
    usage_count = models.IntegerField(default=0)

    # Expiration
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(
        help_text="Key expiration date (default: 90 days from creation)",
    )

    # Rate limiting
    rate_limit_per_minute = models.IntegerField(
        default=60,
        help_text="Max requests per minute (0 = unlimited)",
    )
    rate_limit_per_hour = models.IntegerField(
        default=1000,
        help_text="Max requests per hour (0 = unlimited)",
    )

    # IP restrictions (optional)
    allowed_ips = models.JSONField(
        default=list,
        blank=True,
        help_text='List of allowed IP addresses/CIDRs (empty = allow all)',
    )

    # Key rotation
    version = models.IntegerField(default=1)
    rotated_from = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='rotated_to',
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['key_prefix']),
            models.Index(fields=['organization', 'is_active']),
            models.Index(fields=['hashed_key']),
            models.Index(fields=['api_key_id']),
        ]
        verbose_name = 'API Key'
        verbose_name_plural = 'API Keys'

    def __str__(self):
        return f"{self.name} ({self.key_prefix}...)"

    # ── Key generation & verification ──────────────────────────────

    @classmethod
    def generate_key(cls):
        """
        Generate an API key.
        Returns: (full_key, key_prefix, hashed_key)

        The full_key is returned to the caller ONCE and never stored.
        """
        prefix = getattr(settings, 'API_KEY_PREFIX', 'ph')
        random_part = secrets.token_urlsafe(32)  # 256 bits of entropy
        full_key = f"{prefix}_live_{random_part}"
        display_prefix = full_key[:12]
        hashed = cls.hash_key(full_key)
        return full_key, display_prefix, hashed

    @staticmethod
    def hash_key(key: str) -> str:
        """HMAC-SHA256 hash of an API key using server-side secret."""
        return hmac.new(
            _get_hmac_secret(), key.encode(), hashlib.sha256
        ).hexdigest()

    def verify_key(self, raw_key: str) -> bool:
        """Timing-safe verification of a raw API key against stored hash."""
        return hmac.compare_digest(
            self.hashed_key,
            self.__class__.hash_key(raw_key),
        )

    # ── Validation ─────────────────────────────────────────────────

    def is_valid(self) -> bool:
        """Check if key is active and not expired."""
        if not self.is_active:
            return False
        if self.expires_at and self.expires_at < timezone.now():
            return False
        return True

    def has_permission(self, required_permission: str) -> bool:
        """
        Check if key meets required permission level.
        Hierarchy: admin > write > read
        """
        current = self.PERMISSION_LEVELS.get(self.permissions, 0)
        required = self.PERMISSION_LEVELS.get(required_permission, 0)
        return current >= required

    def has_scope(self, required_scope: str) -> bool:
        """
        Check if key has the required scope.
        Empty scopes list or '*' = allow all.
        Supports hierarchical matching: 'projects:read' matches 'projects:*'.
        """
        if not self.scopes or '*' in self.scopes:
            return True
        if required_scope in self.scopes:
            return True
        # Check wildcard: 'projects:*' matches 'projects:read'
        parts = required_scope.split(':')
        if len(parts) >= 2 and f"{parts[0]}:*" in self.scopes:
            return True
        return False

    def is_ip_allowed(self, ip_address: str) -> bool:
        """
        Check if IP is in the allowed list. Empty list = allow all.
        Supports exact IPs and CIDR notation.
        """
        if not self.allowed_ips:
            return True
        try:
            addr = ipaddress.ip_address(ip_address)
            for allowed in self.allowed_ips:
                if '/' in allowed:
                    if addr in ipaddress.ip_network(allowed, strict=False):
                        return True
                else:
                    if addr == ipaddress.ip_address(allowed):
                        return True
            return False
        except ValueError:
            return False

    # ── Usage tracking ─────────────────────────────────────────────

    def record_usage(self, ip_address: str = None):
        """Atomically increment usage counter and update last_used_at."""
        self.__class__.objects.filter(pk=self.pk).update(
            usage_count=models.F('usage_count') + 1,
            last_used_at=timezone.now(),
        )

    # ── Lifecycle ──────────────────────────────────────────────────

    def revoke(self):
        """Revoke (deactivate) this API key."""
        self.is_active = False
        self.save(update_fields=['is_active'])

    def renew(self, days=90):
        """Extend expiration by specified days."""
        self.expires_at = timezone.now() + timedelta(days=days)
        self.save(update_fields=['expires_at'])

    def save(self, *args, **kwargs):
        if not self.expires_at:
            default_days = getattr(settings, 'API_KEY_DEFAULT_EXPIRY_DAYS', 90)
            self.expires_at = timezone.now() + timedelta(days=default_days)
        super().save(*args, **kwargs)


class APIKeyUsageLog(models.Model):
    """Track API key usage for analytics and auditing."""

    api_key = models.ForeignKey(
        APIKey,
        on_delete=models.CASCADE,
        related_name='usage_logs',
    )

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)

    # Request details
    endpoint = models.CharField(max_length=255)
    method = models.CharField(max_length=10)
    status_code = models.IntegerField()
    response_time_ms = models.IntegerField(null=True)

    # Client info
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
        return f"{self.api_key.key_prefix}... - {self.method} {self.endpoint} ({self.timestamp})"


class APIKeyRateLimit(models.Model):
    """
    DEPRECATED: Rate limiting now uses Redis via APIKeyAuth.check_rate_limit().
    Kept for historical data only — no longer written to during request processing.
    """

    api_key = models.ForeignKey(
        APIKey,
        on_delete=models.CASCADE,
        related_name='rate_limits',
    )

    window_start = models.DateTimeField(db_index=True)
    window_type = models.CharField(
        max_length=10,
        choices=[
            ('minute', 'Per Minute'),
            ('hour', 'Per Hour'),
        ],
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
        return f"{self.api_key.key_prefix}... - {self.window_type} ({self.request_count} requests)"
