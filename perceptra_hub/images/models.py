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
    
    image_name = models.CharField(max_length=255, unique=True)
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
            models.Index(fields=['storage_profile', 'storage_key']),
            models.Index(fields=['uploaded_by', 'created_at']),
        ]
        # Ensure storage_key is unique per organization
        unique_together = [('organization', 'storage_key')]

    def __str__(self):
        return f"{self.name} ({self.organization.name})"
    
    def get_download_url(self, expiration: int = 3600) -> str:
        """
        Generate a presigned download URL for this image.
        
        Args:
            expiration: URL expiration in seconds
        
        Returns:
            Presigned URL string
        """
        from storage.services import get_storage_adapter_for_profile
        
        adapter = get_storage_adapter_for_profile(self.storage_profile)
        presigned = adapter.generate_presigned_url(
            self.storage_key,
            expiration=expiration,
            method='GET'
        )
        return presigned.url
    
    def delete_from_storage(self):
        """Delete the actual file from storage."""
        from storage.services import get_storage_adapter_for_profile
        
        adapter = get_storage_adapter_for_profile(self.storage_profile)
        adapter.delete_file(self.storage_key)
    
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