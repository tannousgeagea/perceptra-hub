from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from django.db.models import Count, Sum
from django.utils.translation import gettext_lazy as _
from .models import Image, Tag, ImageTag
from unfold.admin import ModelAdmin, TabularInline


class ImageTagInline(TabularInline):
    """Inline for managing image tags."""
    model = ImageTag
    extra = 1
    autocomplete_fields = ['tag', 'tagged_by']
    readonly_fields = ['tagged_at']
    fields = ['tag', 'tagged_by', 'tagged_at']


@admin.register(Image)
class ImageAdmin(ModelAdmin):
    """Admin for Image model."""
    
    list_display = [
        'thumbnail_preview',
        'name',
        'organization',
        'dimensions_display',
        'file_size_display',
        'format_display',
        'storage_profile',
        'uploaded_by',
        'created_at',
        'actions_display',
    ]
    
    list_filter = [
        'organization',
        'storage_profile',
        'file_format',
        'source_of_origin',
        'created_at',
        'uploaded_by'
    ]
    
    search_fields = [
        'name',
        'original_filename',
        'image_id',
        'storage_key',
        'checksum'
    ]
    
    readonly_fields = [
        'image_id',
        'thumbnail_large',
        'checksum',
        'file_size',
        'width',
        'height',
        'aspect_ratio',
        'megapixels',
        'file_size_mb',
        'created_at',
        'updated_at',
        'download_link'
    ]
    
    fieldsets = (
        (_('Basic Information'), {
            'fields': (
                'image_id',
                'organization',
                'name',
                'original_filename',
                'thumbnail_large'
            )
        }),
        (_('Storage'), {
            'fields': (
                'storage_profile',
                'storage_key',
                'download_link'
            )
        }),
        (_('File Information'), {
            'fields': (
                'file_format',
                'file_size',
                'file_size_mb',
                'checksum'
            )
        }),
        (_('Dimensions'), {
            'fields': (
                'width',
                'height',
                'aspect_ratio',
                'megapixels'
            )
        }),
        (_('Origin & Metadata'), {
            'fields': (
                'source_of_origin',
                'uploaded_by',
                'meta_info'
            )
        }),
        (_('Timestamps'), {
            'fields': (
                'created_at',
                'updated_at'
            ),
            'classes': ('collapse',)
        })
    )
    
    inlines = [ImageTagInline]
    
    autocomplete_fields = ['organization', 'storage_profile', 'uploaded_by']
    
    date_hierarchy = 'created_at'
    
    list_per_page = 50
    
    actions = ['generate_download_urls', 'verify_checksums']
    
    def thumbnail_preview(self, obj):
        """Small thumbnail for list view."""
        try:
            url = obj.get_download_url(expiration=300)
            return format_html(
                '<img src="{}" style="width: 50px; height: 50px; object-fit: cover; border-radius: 4px;" />',
                url
            )
        except Exception:
            return format_html(
                '<div style="width: 50px; height: 50px; background: #ddd; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px;">N/A</div>'
            )
    thumbnail_preview.short_description = _('Preview')
    
    def thumbnail_large(self, obj):
        """Large thumbnail for detail view."""
        try:
            url = obj.get_download_url(expiration=300)
            return format_html(
                '<img src="{}" style="max-width: 400px; max-height: 400px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />',
                url
            )
        except Exception as e:
            return format_html(
                '<div style="padding: 20px; background: #f8f9fa; border-radius: 8px; color: #6c757d;">Image preview unavailable: {}</div>',
                str(e)
            )
    thumbnail_large.short_description = _('Image Preview')
    
    def dimensions_display(self, obj):
        """Display dimensions with aspect ratio."""
        return format_html(
            '<strong>{}×{}</strong><br/><small>{}:1 • {}MP</small>',
            obj.width,
            obj.height,
            f'{obj.aspect_ratio:.2f}',
            f'{obj.megapixels:.1f}'
        )
    dimensions_display.short_description = _('Dimensions')
    
    def file_size_display(self, obj):
        """Display file size in human-readable format."""
        size_mb = obj.file_size_mb
        if size_mb < 1:
            size_kb = obj.file_size / 1024
            return format_html('<strong>{}</strong> KB', f'{size_kb:.1f}')
        return format_html('<strong>{}</strong> MB', f'{size_mb:.2f}')
    file_size_display.short_description = _('Size')
    file_size_display.admin_order_field = 'file_size'
    
    def format_display(self, obj):
        """Display file format with color coding."""
        colors = {
            'jpg': '#10b981',
            'jpeg': '#10b981',
            'png': '#3b82f6',
            'tiff': '#8b5cf6',
            'tif': '#8b5cf6',
            'bmp': '#f59e0b',
            'webp': '#ec4899'
        }
        color = colors.get(obj.file_format.lower(), '#6b7280')
        return format_html(
            '<span style="background: {}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; text-transform: uppercase;">{}</span>',
            color,
            obj.file_format
        )
    format_display.short_description = _('Format')
    format_display.admin_order_field = 'file_format'
    
    def download_link(self, obj):
        """Generate download link."""
        try:
            url = obj.get_download_url(expiration=3600)
            return format_html(
                '<a href="{}" target="_blank" style="display: inline-block; padding: 8px 16px; background: #3b82f6; color: white; text-decoration: none; border-radius: 6px; font-weight: 500;">📥 Download Image</a>',
                url
            )
        except Exception as e:
            return format_html(
                '<span style="color: #ef4444;">Download unavailable: {}</span>',
                str(e)
            )
    download_link.short_description = _('Download')
    
    def actions_display(self, obj):
        """Action buttons for list view."""
        return format_html(
            '<a href="{}" style="margin-right: 5px;">📝</a>'
            '<a href="{}" onclick="return confirm(\'Are you sure?\')">🗑️</a>',
            reverse('admin:images_image_change', args=[obj.pk]),
            reverse('admin:images_image_delete', args=[obj.pk])
        )
    actions_display.short_description = _('Actions')
    
    def generate_download_urls(self, request, queryset):
        """Action to generate download URLs."""
        urls = []
        for image in queryset:
            try:
                url = image.get_download_url(expiration=86400)  # 24 hours
                urls.append(f"{image.name}: {url}")
            except Exception as e:
                urls.append(f"{image.name}: Error - {str(e)}")
        
        self.message_user(
            request,
            mark_safe("<br/>".join(urls))
        )
    generate_download_urls.short_description = _('Generate download URLs (24h)')
    
    def verify_checksums(self, request, queryset):
        """Action to verify image checksums."""
        verified = 0
        failed = 0
        
        for image in queryset:
            if image.checksum:
                verified += 1
            else:
                failed += 1
        
        self.message_user(
            request,
            f"Verified {verified} images with checksums. {failed} images without checksums."
        )
    verify_checksums.short_description = _('Verify checksums')
    
    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        qs = super().get_queryset(request)
        return qs.select_related(
            'organization',
            'storage_profile',
            'uploaded_by'
        ).prefetch_related('tags')


@admin.register(Tag)
class TagAdmin(ModelAdmin):
    """Admin for Tag model."""
    
    list_display = [
        'color_badge',
        'name',
        'organization',
        'image_count',
        'description_short',
        'created_at'
    ]
    
    list_filter = [
        'organization',
        'created_at'
    ]
    
    search_fields = [
        'name',
        'description'
    ]
    
    readonly_fields = [
        'created_at',
        'image_count_detail'
    ]
    
    fieldsets = (
        (_('Basic Information'), {
            'fields': (
                'organization',
                'name',
                'description',
                'color'
            )
        }),
        (_('Statistics'), {
            'fields': (
                'image_count_detail',
                'created_at'
            )
        })
    )
    
    autocomplete_fields = ['organization']
    
    date_hierarchy = 'created_at'
    
    list_per_page = 100
    
    def color_badge(self, obj):
        """Display colored badge."""
        return format_html(
            '<span style="display: inline-block; width: 30px; height: 30px; background: {}; border-radius: 50%; border: 2px solid #e5e7eb;"></span>',
            obj.color
        )
    color_badge.short_description = _('Color')
    
    def image_count(self, obj):
        """Count of images with this tag."""
        count = obj.tagged_images.count()
        return format_html(
            '<strong style="color: #3b82f6;">{}</strong> images',
            count
        )
    image_count.short_description = _('Images')
    
    def image_count_detail(self, obj):
        """Detailed image count for detail view."""
        count = obj.tagged_images.count()
        return format_html(
            '<div style="padding: 12px; background: #f8f9fa; border-radius: 8px;">'
            '<strong style="font-size: 24px; color: #3b82f6;">{}</strong> '
            '<span style="color: #6b7280;">images tagged</span>'
            '</div>',
            count
        )
    image_count_detail.short_description = _('Total Images')
    
    def description_short(self, obj):
        """Truncated description."""
        if obj.description:
            return obj.description[:50] + '...' if len(obj.description) > 50 else obj.description
        return format_html('<span style="color: #9ca3af;">No description</span>')
    description_short.short_description = _('Description')
    
    def get_queryset(self, request):
        """Optimize queryset with annotations."""
        qs = super().get_queryset(request)
        return qs.select_related('organization').annotate(
            num_images=Count('tagged_images')
        )


@admin.register(ImageTag)
class ImageTagAdmin(ModelAdmin):
    """Admin for ImageTag relationship model."""
    
    list_display = [
        'image_thumbnail',
        'image_name',
        'tag_display',
        'tagged_by',
        'tagged_at'
    ]
    
    list_filter = [
        'tag',
        'tagged_at',
        'tagged_by'
    ]
    
    search_fields = [
        'image__name',
        'image__original_filename',
        'tag__name'
    ]
    
    readonly_fields = [
        'tagged_at',
        'image_preview'
    ]
    
    autocomplete_fields = ['image', 'tag', 'tagged_by']
    
    date_hierarchy = 'tagged_at'
    
    list_per_page = 100
    
    fieldsets = (
        (_('Relationship'), {
            'fields': (
                'image',
                'image_preview',
                'tag'
            )
        }),
        (_('Metadata'), {
            'fields': (
                'tagged_by',
                'tagged_at'
            )
        })
    )
    
    def image_thumbnail(self, obj):
        """Small thumbnail."""
        try:
            url = obj.image.get_download_url(expiration=300)
            return format_html(
                '<img src="{}" style="width: 40px; height: 40px; object-fit: cover; border-radius: 4px;" />',
                url
            )
        except Exception:
            return '—'
    image_thumbnail.short_description = _('Preview')
    
    def image_name(self, obj):
        """Display image name with link."""
        url = reverse('admin:images_image_change', args=[obj.image.pk])
        return format_html(
            '<a href="{}">{}</a>',
            url,
            obj.image.name
        )
    image_name.short_description = _('Image')
    image_name.admin_order_field = 'image__name'
    
    def tag_display(self, obj):
        """Display tag with color."""
        return format_html(
            '<span style="display: inline-flex; align-items: center; gap: 6px;">'
            '<span style="width: 12px; height: 12px; background: {}; border-radius: 50%;"></span>'
            '<strong>{}</strong>'
            '</span>',
            obj.tag.color,
            obj.tag.name
        )
    tag_display.short_description = _('Tag')
    tag_display.admin_order_field = 'tag__name'
    
    def image_preview(self, obj):
        """Large preview for detail view."""
        try:
            url = obj.image.get_download_url(expiration=300)
            return format_html(
                '<img src="{}" style="max-width: 300px; max-height: 300px; border-radius: 8px;" />',
                url
            )
        except Exception:
            return '—'
    image_preview.short_description = _('Image Preview')
    
    def get_queryset(self, request):
        """Optimize queryset."""
        qs = super().get_queryset(request)
        return qs.select_related(
            'image',
            'image__organization',
            'image__storage_profile',
            'tag',
            'tag__organization',
            'tagged_by'
        )