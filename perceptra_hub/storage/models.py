"""
Storage management models for multi-tenant computer vision platform.
"""
import uuid
from django.db import models
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from organizations.models import Organization

from django.contrib.auth import get_user_model
User = get_user_model()


class TimeStampedModel(models.Model):
    """Abstract model to track creation and modification times and users."""
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="%(class)s_created",
        help_text=_("User who created this record")
    )
    updated_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="%(class)s_updated",
        help_text=_("User who last updated this record")
    )

    class Meta:
        abstract = True


class StorageBackend(models.TextChoices):
    """Supported storage backend types."""
    AZURE = 'azure', _('Azure Blob Storage')
    S3 = 's3', _('Amazon S3')
    MINIO = 'minio', _('MinIO / S3-compatible')
    LOCAL = 'local', _('Local Filesystem')


class SecretProvider(models.TextChoices):
    """Secret management provider types."""
    VAULT = 'vault', _('HashiCorp Vault')
    AZURE_KV = 'azure_kv', _('Azure Key Vault')
    AWS_SM = 'aws_sm', _('AWS Secrets Manager')
    LOCAL_ENC = 'local_enc', _('Local Encrypted Storage')


class SecretRef(models.Model):
    """
    Reference to externally stored secrets.
    
    This model doesn't store actual credentials but references where
    they are stored in external secret management systems.
    """
    credential_ref_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False
    )
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='secret_refs'
    )
    provider = models.CharField(
        max_length=20,
        choices=SecretProvider.choices,
        help_text=_('Secret management provider')
    )
    path = models.CharField(
        max_length=500,
        help_text=_('Path or identifier in the secret provider')
    )
    key = models.CharField(
        max_length=100,
        help_text=_('Specific key/field name within the secret')
    )
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Additional metadata for secret retrieval')
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'storage_secret_ref'
        verbose_name = _('Secret Reference')
        verbose_name_plural = _('Secret References')
        indexes = [
            models.Index(fields=['organization', 'provider']),
        ]

    def __str__(self):
       return f"{self.provider}:{self.path}/{self.key} ({self.organization.name})"

class EncryptedSecret(TimeStampedModel):
    """
    Stores encrypted secrets locally.
    
    SECURITY: The encryption key must NEVER be stored in the database.
    It should be in settings.SECRET_ENCRYPTION_KEY from environment variables.
    """
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='encrypted_secrets'
    )
    identifier = models.CharField(
        max_length=255,
        help_text=_('Unique identifier for this secret')
    )
    encrypted_value = models.TextField(
        help_text=_('Fernet-encrypted secret data')
    )
    description = models.TextField(
        blank=True,
        help_text=_('Description of what this secret is for')
    )
    
    encryption_version = models.PositiveSmallIntegerField(
        default=1,
        help_text=_('Encryption key version used for this secret')
    )
    last_decrypted_at = models.DateTimeField(
        null=True, blank=True,
        help_text=_('Timestamp of last successful decryption (for audit)')
    )
    
    class Meta:
        db_table = 'storage_encrypted_secret'
        verbose_name = _('Encrypted Secret')
        verbose_name_plural = _('Encrypted Secrets')
        unique_together = [('organization', 'identifier')]
        indexes = [
            models.Index(fields=['organization', 'identifier']),
        ]
    
    def __str__(self):
        return f"{self.identifier} ({self.organization.name})"

    def get_decrypted_value(self, encryption_key: str | bytes) -> dict:
        """
        Decrypt and return the secret value as a dictionary.
        Does NOT persist 'last_decrypted_at'; call update_last_decrypted() if desired.
        """
        from cryptography.fernet import Fernet, InvalidToken
        import json

        try:
            fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
            decrypted = fernet.decrypt(self.encrypted_value.encode()).decode('utf-8')
            return json.loads(decrypted)
        except InvalidToken as e:
            raise ValueError("Invalid encryption key or corrupted secret.") from e
        except json.JSONDecodeError:
            return {"value": decrypted}

    def update_last_decrypted(self, commit: bool = True) -> None:
        """Update timestamp when secret is successfully decrypted."""
        from django.utils import timezone
        self.last_decrypted_at = timezone.now()
        if commit:
            self.save(update_fields=['last_decrypted_at'])

class StorageProfile(models.Model):
    """
    Storage configuration profile for organization data and models.
    
    Each organization can have multiple storage profiles for different purposes
    (e.g., raw data, processed data, models). One profile can be marked
    as default per organization.
    """
    storage_profile_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False
    )
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='storage_profiles'
    )
    name = models.CharField(
        max_length=100,
        help_text=_('Human-readable profile name')
    )
    backend = models.CharField(
        max_length=20,
        choices=StorageBackend.choices,
        help_text=_('Storage backend type')
    )
    region = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text=_('Region or location (for cloud providers)')
    )
    is_default = models.BooleanField(
        default=False,
        help_text=_('Whether this is the default storage profile for the tenant')
    )
    config = models.JSONField(
        default=dict,
        help_text=_(
            'Backend-specific configuration (excluding secrets). '
            'Examples: bucket_name, container_name, base_path, endpoint_url'
        )
    )
    credential_ref = models.ForeignKey(
        SecretRef,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='storage_profiles',
        help_text=_('Reference to stored credentials')
    )
    is_active = models.BooleanField(
        default=True,
        help_text=_('Whether this profile is currently active')
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'storage_profile'
        verbose_name = _('Storage Profile')
        verbose_name_plural = _('Storage Profiles')
        unique_together = [('organization', 'name')]
        indexes = [
            models.Index(fields=['organization', 'is_default']),
            models.Index(fields=['organization', 'backend']),
            models.Index(fields=['is_active']),
        ]

    def __str__(self):
        return f"{self.organization.name} - {self.name} ({self.backend})"

    def clean(self):
        """Validate storage profile configuration."""
        super().clean()
        
        # Validate required config fields based on backend
        required_fields = self._get_required_config_fields()
        missing_fields = [
            field for field in required_fields 
            if field not in self.config
        ]
        
        if missing_fields:
            raise ValidationError({
                'config': _(
                    f"Missing required fields for {self.backend}: "
                    f"{', '.join(missing_fields)}"
                )
            })
        
        # Validate credential_ref belongs to same organization
        if self.credential_ref and self.credential_ref.organization != self.organization:
            raise ValidationError({
                'credential_ref': _('Credential reference must belong to the same organization')
            })
        
        # Validate only one default per organization
        if self.is_default:
            existing_default = StorageProfile.objects.filter(
                organization=self.organization,
                is_default=True
            ).exclude(pk=self.pk).first()
            
            if existing_default:
                raise ValidationError({
                    'is_default': _(
                        f"Organization already has a default storage profile: "
                        f"{existing_default.name}"
                    )
                })

    def _get_required_config_fields(self) -> list[str]:
        """Get required config fields based on backend type."""
        backend_requirements = {
            StorageBackend.AZURE: ['container_name', 'account_name'],
            StorageBackend.S3: ['bucket_name'],
            StorageBackend.MINIO: ['bucket_name', 'endpoint_url'],
            StorageBackend.LOCAL: ['base_path'],
        }
        return backend_requirements.get(self.backend, [])

    def save(self, *args, **kwargs):
        """Override save to run validation."""
        self.full_clean()
        super().save(*args, **kwargs)

    @property
    def requires_credentials(self) -> bool:
        """Check if this backend requires credentials."""
        return self.backend != StorageBackend.LOCAL

    def get_config_display(self) -> dict:
        """
        Get sanitized config for display purposes.
        Removes any potentially sensitive information.
        """
        safe_config = self.config.copy()
        sensitive_keys = ['access_key', 'secret_key', 'token', 'password']
        
        for key in sensitive_keys:
            if key in safe_config:
                safe_config[key] = '***REDACTED***'
        
        return safe_config