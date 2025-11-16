
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
import uuid

User = get_user_model()


class OAuthProvider(models.TextChoices):
    """OAuth provider choices."""
    MICROSOFT = 'microsoft', 'Microsoft'
    GOOGLE = 'google', 'Google'


class SocialAccount(models.Model):
    """
    Links user accounts with external OAuth providers.
    Stores OAuth tokens and provider-specific user IDs.
    """
    uuid = models.UUIDField(default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='social_accounts'
    )
    provider = models.CharField(
        max_length=20,
        choices=OAuthProvider.choices
    )
    provider_user_id = models.CharField(
        max_length=255,
        help_text="User ID from the OAuth provider"
    )

    # Stable unique identifier from the provider – usually the 'sub' claim.
    subject = models.CharField(
        max_length=255,
        help_text="Stable subject/identifier from the provider (e.g. 'sub' claim).",
    )

    # OIDC issuer (iss) – important if you later support multiple issuers per provider.
    issuer = models.CharField(
        max_length=255,
        blank=True,
        help_text="OIDC issuer URI / authority (e.g. https://login.microsoftonline.com/{tenant}/v2.0).",
    )

    # Provider-side tenant id (e.g. Azure AD 'tid'), if applicable.
    tenant_id = models.CharField(
        max_length=128,
        blank=True,
        help_text="Provider tenant identifier (e.g. Azure AD tenant ID).",
    )
    email = models.EmailField(
        help_text="Email from OAuth provider"
    )
    email_verified = models.BooleanField(
        default=False,
        help_text="Whether the provider marked this email as verified.",
    )
    display_name = models.CharField(
        max_length=255,
        blank=True,
        help_text="Full display name from the provider profile.",
    )
    given_name = models.CharField(
        max_length=150,
        blank=True,
    )
    family_name = models.CharField(
        max_length=150,
        blank=True,
    )
    avatar_url = models.URLField(
        max_length=500,
        blank=True,
        help_text="Profile picture URL if provided.",
    )

    # Flags & auditing
    is_primary = models.BooleanField(
        default=False,
        help_text="Whether this is the primary external account for this provider.",
    )
    is_revoked = models.BooleanField(
        default=False,
        help_text="Set to True if user revoked / unlinked this external account.",
    )
    
    # OAuth tokens (optional - store if you need to call provider APIs later)
    access_token = models.TextField(blank=True, null=True)
    refresh_token = models.TextField(blank=True, null=True)
    token_expires_at = models.DateTimeField(blank=True, null=True)
    
    # Metadata
    extra_data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Additional data from provider (name, avatar, etc.)"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_login = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = 'social_accounts'
        unique_together = [['provider', 'provider_user_id']]
        indexes = [
            models.Index(fields=['user', 'provider']),
            models.Index(fields=['provider', 'email']),
            models.Index(fields=["provider", "subject"], name="idx_extacc_provider_subject"),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.provider}"


    @property
    def is_active(self) -> bool:
        return not self.is_revoked

    def revoke(self, save: bool = True) -> None:
        """
        Logically unlink this external account without deleting the row.
        Keeps history but prevents further sign-in using it.
        """
        self.is_revoked = True
        if save:
            self.save(update_fields=["is_revoked"])
