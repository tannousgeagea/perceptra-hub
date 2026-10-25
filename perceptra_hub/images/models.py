from django.db import models
from tenants.models import EdgeBox, SensorBox
from django.core.validators import MinValueValidator
from django.utils.translation import gettext_lazy as _
from django.contrib.auth import get_user_model
from organizations.models import Organization
from storage.models import StorageProfile

User = get_user_model()

# Create your models here.
class Image(models.Model):
    """
    Image model for storing image metadata and storage location.
    
    Images are stored in organization's storage profiles, not Django media.
    """
    image_id = models.CharField(max_length=255, unique=True)

    # Identity in the producing system, for images registered by reference
    # (POST /api/v1/ingest/images): re-registering the same external_id
    # updates this image instead of creating another.
    external_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text=_('Identifier in the system that registered this image (idempotency key)')
    )

    # Organization relationship (IMPORTANT!)
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='images',
        help_text=_('Organization that owns this image')
    )
    
    # Storage information (CRITICAL for multi-cloud storage)
    storage_profile = models.ForeignKey(
        StorageProfile,
        on_delete=models.PROTECT,  # Prevent deletion of profile with images
        related_name='images',
        help_text=_('Storage profile where this image is stored')
    )
    storage_key = models.CharField(
        max_length=500,
        help_text=_('Full path/key in storage (e.g., org-123/images/2025/img.jpg)')
    )
    
    # Image metadata
    name = models.CharField(
        max_length=255,
        help_text=_('Human-readable image name')
    )
    original_filename = models.CharField(
        max_length=255,
        help_text=_('Original filename when uploaded')
    )
    file_format = models.CharField(
        max_length=10,
        help_text=_('Image format (jpg, png, tiff, etc.)')
    )
    file_size = models.BigIntegerField(
        validators=[MinValueValidator(0)],
        help_text=_('File size in bytes')
    )
    
    # Image dimensions
    width = models.PositiveIntegerField(
        blank=True,
        help_text=_('Image width in pixels')
    )
    height = models.PositiveIntegerField(
        blank=True,
        help_text=_('Image height in pixels')
    )
    
    image_name = models.CharField(max_length=255)
    image_file = models.ImageField(upload_to='images/')
    
    # Origin tracking
    source_of_origin = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text=_('Source where image came from (camera, upload, API, etc.)')
    )
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='uploaded_images',
        help_text=_('User who uploaded this image')
    )
    
    # Additional metadata
    meta_info = models.JSONField(
        null=True,
        blank=True,
        default=dict,
        help_text=_('Additional metadata (EXIF, GPS, camera info, etc.)')
    )
    
    # Checksum for integrity verification
    checksum = models.CharField(
        max_length=64,
        null=True,
        blank=True,
        help_text=_('SHA-256 checksum of file')
    )
    
    tags = models.ManyToManyField(
        'Tag',
        through='ImageTag',
        related_name='images',
        blank=True
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'image'
        verbose_name = _('Image')
        verbose_name_plural = _('Images')
        indexes = [
            models.Index(fields=['organization', 'created_at']),
            models.Index(fields=['uploaded_by', 'created_at']),
        ]
        constraints = [
            # A key is unique within its storage: several external storages
            # of one organization may well use the same paths.
            models.UniqueConstraint(fields=['storage_profile', 'storage_key'], name='uniq_image_profile_key'),
            models.UniqueConstraint(
                fields=['organization', 'external_id'], name='uniq_image_org_external_id',
                condition=models.Q(external_id__isnull=False),
            ),
        ]

    def __str__(self):
        return f"{self.name} ({self.organization.name})"
    
    def get_download_url(self, expiration: int = 3600) -> str:
        """
        Generate a short-lived download URL for this image.

        The backend's own presigned URL when it can mint one; otherwise
        (e.g. an Azure SAS-only credential) a signed URL of Perceptra's media
        proxy, so a long-lived storage token never reaches a browser.

        Args:
            expiration: URL expiration in seconds

        Returns:
            URL string
        """
        from storage.services import get_storage_adapter_for_profile

        if self.storage_profile.backend == "local":
            from api.routers.images.queries.media import _media_url
            return _media_url(
                storage_backend='local',
                storage_key=f"{self.storage_profile.config['base_path']}/{self.storage_key}"
            )

        adapter = get_storage_adapter_for_profile(self.storage_profile)
        if not adapter.can_presign:
            return self.get_proxy_url(expiration)
        presigned = adapter.generate_presigned_url(
            self.storage_key,
            expiration=expiration,
            method='GET'
        )
        return presigned.url

    def get_proxy_url(self, expiration: int = 3600) -> str:
        """Signed URL of the media proxy streaming this image (no auth header needed)."""
        from django.conf import settings
        from storage.signing import signed_query

        path = f"/api/v1/media/images/{self.image_id}"
        return f"{getattr(settings, 'MEDIA_BASE_URL', '')}{path}?{signed_query(path, expiration)}"

    @property
    def is_external(self) -> bool:
        """The file belongs to another system (registered by reference)."""
        return self.storage_profile.is_external

    def delete_from_storage(self) -> bool:
        """
        Delete the actual file from storage. A no-op for external storage:
        its files belong to another system, deleting the image only unlinks
        it. Returns whether a file was deleted.
        """
        from storage.services import get_storage_adapter_for_profile

        if self.storage_profile.is_external:
            return False
        adapter = get_storage_adapter_for_profile(self.storage_profile)
        adapter.delete_file(self.storage_key)
        return True
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate image aspect ratio."""
        if self.height > 0:
            return self.width / self.height
        return 0.0
    
    @property
    def megapixels(self) -> float:
        """Calculate megapixels."""
        return (self.width * self.height) / 1_000_000
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size / (1024 * 1024)
    
class Tag(models.Model):
    # Organization relationship
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='tags',
        help_text=_('Organization that owns this tag')
    )
    
    name = models.CharField(
        max_length=100,
        help_text=_('Tag name')
    )
    
    description = models.TextField(
        blank=True,
        help_text=_('Tag description')
    )
    
    color = models.CharField(
        max_length=7,
        default='#3B82F6',
        help_text=_('Hex color for UI display')
    )
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'tag'
        verbose_name = _('Tag')
        verbose_name_plural = _('Tags')
        unique_together = [('organization', 'name')]
        indexes = [
            models.Index(fields=['organization', 'name']),
        ]

    def __str__(self):
        return f"{self.name} ({self.organization.name})"

class ImageTag(models.Model):
    image = models.ForeignKey(Image, on_delete=models.CASCADE, related_name='image_tags')
    tag = models.ForeignKey(Tag, on_delete=models.CASCADE, related_name='tagged_images')
    tagged_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    tagged_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('image', 'tag')
        db_table = 'image_tag'
        verbose_name_plural = 'Image Tags'

    def __str__(self):
        return f"{self.image.image_name} - {self.tag.name}"