
# Register your models here.
from django.contrib import admin
from unfold.admin import ModelAdmin, TabularInline
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _
from django.db.models import Count, Q
from django.urls import reverse
from django.utils import timezone

from .models import (
    ProjectType, 
    Visibility, 
    Project, 
    ProjectMetadata, 
    ProjectImage, 
    ImageMode,
    Version,
    VersionImage,
)



@admin.register(ProjectType)
class ProjectTypeAdmin(ModelAdmin):
    """Admin interface for project types."""
    
    list_display = [
        'name',
        'description_preview',
        'project_count',
        'is_active',
        'created_at'
    ]
    
    list_filter = [
        'is_active',
        'created_at'
    ]
    
    search_fields = [
        'name',
        'description'
    ]
    
    readonly_fields = [
        'created_at',
        'updated_at',
        'created_by',
        'updated_by',
        'active_project_count'
    ]
    
    fieldsets = (
        (_('Basic Information'), {
            'fields': ('name', 'description', 'is_active')
        }),
        (_('Configuration'), {
            'fields': ('meta_info',),
            'description': _('Additional metadata for project type configuration (JSON format)')
        }),
        (_('Audit'), {
            'fields': ('created_at', 'updated_at', 'created_by', 'updated_by', 'active_project_count'),
            'classes': ('collapse',)
        }),
    )
    
    def description_preview(self, obj):
        """Show truncated description."""
        if obj.description:
            return obj.description[:60] + '...' if len(obj.description) > 60 else obj.description
        return '-'
    description_preview.short_description = _('Description')
    
    def project_count(self, obj):
        """Show count of projects using this type."""
        count = obj.projects.count()
        if count > 0:
            url = reverse('admin:projects_project_changelist') + f'?project_type__id__exact={obj.id}'
            return format_html(
                '<a href="{}" style="color: #417690; font-weight: bold;">{} project(s)</a>',
                url, count
            )
        return format_html('<span style="color: gray;">0</span>')
    project_count.short_description = _('Projects')
    
    def active_project_count(self, obj):
        """Show detailed project counts."""
        if not obj.pk:
            return '-'
        
        total = obj.projects.count()
        active = obj.projects.filter(is_active=True, is_deleted=False).count()
        archived = obj.projects.filter(is_active=False, is_deleted=False).count()
        deleted = obj.projects.filter(is_deleted=True).count()
        
        return format_html(
            '<ul style="margin: 0; padding-left: 20px;">'
            '<li>Total: <strong>{}</strong></li>'
            '<li>Active: <strong style="color: green;">{}</strong></li>'
            '<li>Archived: <strong style="color: orange;">{}</strong></li>'
            '<li>Deleted: <strong style="color: red;">{}</strong></li>'
            '</ul>',
            total, active, archived, deleted
        )
    active_project_count.short_description = _('Project Statistics')
    
    def save_model(self, request, obj, form, change):
        """Set created_by and updated_by."""
        if not change:
            obj.created_by = request.user
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)
    
    def get_queryset(self, request):
        """Annotate queryset with project counts."""
        qs = super().get_queryset(request)
        return qs.annotate(
            _project_count=Count('projects')
        )


@admin.register(Visibility)
class VisibilityAdmin(ModelAdmin):
    """Admin interface for visibility options."""
    
    list_display = [
        # 'display_order',
        'name',
        'description_preview',
        'project_count',
        'created_at'
    ]
    
    # list_editable = ['display_order']
    
    list_filter = [
        'created_at'
    ]
    
    search_fields = [
        'name',
        'description'
    ]
    
    readonly_fields = [
        'created_at',
        'updated_at',
        'created_by',
        'updated_by'
    ]
    
    fieldsets = (
        (_('Basic Information'), {
            'fields': ('name', 'description', 'display_order')
        }),
        (_('Audit'), {
            'fields': ('created_at', 'updated_at', 'created_by', 'updated_by'),
            'classes': ('collapse',)
        }),
    )
    
    def description_preview(self, obj):
        """Show truncated description."""
        if obj.description:
            return obj.description[:50] + '...' if len(obj.description) > 50 else obj.description
        return '-'
    description_preview.short_description = _('Description')
    
    def project_count(self, obj):
        """Show count of projects with this visibility."""
        count = obj.projects.count()
        if count > 0:
            url = reverse('admin:projects_project_changelist') + f'?visibility__id__exact={obj.id}'
            return format_html(
                '<a href="{}" style="color: #417690; font-weight: bold;">{} project(s)</a>',
                url, count
            )
        return format_html('<span style="color: gray;">0</span>')
    project_count.short_description = _('Projects')
    
    def save_model(self, request, obj, form, change):
        """Set created_by and updated_by."""
        if not change:
            obj.created_by = request.user
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)


class ProjectStatusFilter(admin.SimpleListFilter):
    """Custom filter for project status."""
    title = _('Status')
    parameter_name = 'status'
    
    def lookups(self, request, model_admin):
        return (
            ('active', _('Active')),
            ('archived', _('Archived')),
            ('deleted', _('Deleted')),
        )
    
    def queryset(self, request, queryset):
        if self.value() == 'active':
            return queryset.filter(is_active=True, is_deleted=False)
        if self.value() == 'archived':
            return queryset.filter(is_active=False, is_deleted=False)
        if self.value() == 'deleted':
            return queryset.filter(is_deleted=True)
        return queryset


@admin.register(Project)
class ProjectAdmin(ModelAdmin):
    """Admin interface for projects."""
    
    list_display = [
        'name',
        'organization',
        'project_type',
        'visibility',
        'status_badge',
        'last_edited',
        'created_at'
    ]
    
    list_filter = [
        ProjectStatusFilter,
        'project_type',
        'visibility',
        'organization',
        'created_at',
        'last_edited'
    ]
    
    search_fields = [
        'name',
        'description',
        'project_id',
        'organization__name'
    ]
    
    readonly_fields = [
        'project_id',
        'created_at',
        'updated_at',
        'last_edited',
        'created_by',
        'updated_by',
        'deleted_at',
        'deleted_by',
        'thumbnail_preview',
        'status_info'
    ]
    
    fieldsets = (
        (_('Basic Information'), {
            'fields': (
                'project_id',
                'organization',
                'name',
                'description',
                'thumbnail_url',
                'thumbnail_preview'
            )
        }),
        (_('Configuration'), {
            'fields': (
                'project_type',
                'visibility',
                'settings'
            )
        }),
        (_('Status'), {
            'fields': (
                'is_active',
                'is_deleted',
                'status_info'
            )
        }),
        (_('Audit Information'), {
            'fields': (
                'created_at',
                'created_by',
                'updated_at',
                'updated_by',
                'last_edited',
                'deleted_at',
                'deleted_by'
            ),
            'classes': ('collapse',)
        }),
    )
    
    actions = [
        'archive_projects',
        'activate_projects',
        'soft_delete_projects',
        'restore_projects'
    ]
    
    def status_badge(self, obj):
        """Display colored status badge."""
        if obj.is_deleted:
            return format_html(
                '<span style="background: #dc3545; color: white; padding: 3px 8px; '
                'border-radius: 3px; font-size: 11px; font-weight: bold;">DELETED</span>'
            )
        elif obj.is_active:
            return format_html(
                '<span style="background: #28a745; color: white; padding: 3px 8px; '
                'border-radius: 3px; font-size: 11px; font-weight: bold;">ACTIVE</span>'
            )
        else:
            return format_html(
                '<span style="background: #ffc107; color: #333; padding: 3px 8px; '
                'border-radius: 3px; font-size: 11px; font-weight: bold;">ARCHIVED</span>'
            )
    status_badge.short_description = _('Status')
    
    def thumbnail_preview(self, obj):
        """Show thumbnail preview if URL exists."""
        if obj.thumbnail_url:
            return format_html(
                '<img src="{}" style="max-width: 200px; max-height: 200px; '
                'border: 1px solid #ddd; border-radius: 4px; padding: 5px;" />',
                obj.thumbnail_url
            )
        return _('No thumbnail')
    thumbnail_preview.short_description = _('Thumbnail Preview')
    
    def status_info(self, obj):
        """Show detailed status information."""
        if not obj.pk:
            return '-'
        
        info = []
        
        if obj.is_deleted:
            info.append(format_html(
                '<div style="padding: 10px; background: #f8d7da; border-left: 4px solid #dc3545; margin-bottom: 10px;">'
                '<strong>Deleted</strong><br>'
                'At: {}<br>By: {}'
                '</div>',
                obj.deleted_at.strftime('%Y-%m-%d %H:%M:%S') if obj.deleted_at else 'Unknown',
                obj.deleted_by.get_full_name() if obj.deleted_by else 'Unknown'
            ))
        elif obj.is_archived:
            info.append(format_html(
                '<div style="padding: 10px; background: #fff3cd; border-left: 4px solid #ffc107; margin-bottom: 10px;">'
                '<strong>Archived</strong><br>'
                'Last edited: {}'
                '</div>',
                obj.last_edited.strftime('%Y-%m-%d %H:%M:%S')
            ))
        else:
            info.append(format_html(
                '<div style="padding: 10px; background: #d4edda; border-left: 4px solid #28a745; margin-bottom: 10px;">'
                '<strong>Active</strong><br>'
                'Last edited: {}'
                '</div>',
                obj.last_edited.strftime('%Y-%m-%d %H:%M:%S')
            ))
        
        return format_html(''.join([str(i) for i in info]))
    status_info.short_description = _('Status Details')
    
    def get_readonly_fields(self, request, obj=None):
        """Make certain fields readonly for deleted projects."""
        readonly = list(super().get_readonly_fields(request, obj))
        
        if obj and obj.is_deleted:
            # Make most fields readonly for deleted projects
            readonly.extend(['name', 'description', 'project_type', 
                           'visibility', 'organization', 'settings'])
        
        return readonly
    
    def save_model(self, request, obj, form, change):
        """Set created_by and updated_by."""
        if not change:
            obj.created_by = request.user
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)
    
    # Admin Actions
    def archive_projects(self, request, queryset):
        """Archive selected projects."""
        count = queryset.filter(is_deleted=False).update(is_active=False)
        self.message_user(
            request,
            _(f'{count} project(s) archived successfully.')
        )
    archive_projects.short_description = _('Archive selected projects')
    
    def activate_projects(self, request, queryset):
        """Activate selected projects."""
        count = queryset.filter(is_deleted=False).update(is_active=True)
        self.message_user(
            request,
            _(f'{count} project(s) activated successfully.')
        )
    activate_projects.short_description = _('Activate selected projects')
    
    def soft_delete_projects(self, request, queryset):
        """Soft delete selected projects."""
        count = 0
        for project in queryset.filter(is_deleted=False):
            project.soft_delete(user=request.user)
            count += 1
        
        self.message_user(
            request,
            _(f'{count} project(s) deleted successfully.')
        )
    soft_delete_projects.short_description = _('Delete selected projects')
    
    def restore_projects(self, request, queryset):
        """Restore soft-deleted projects."""
        count = 0
        for project in queryset.filter(is_deleted=True):
            project.restore()
            count += 1
        
        self.message_user(
            request,
            _(f'{count} project(s) restored successfully.')
        )
    restore_projects.short_description = _('Restore selected projects')
    
    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        qs = super().get_queryset(request)
        return qs.select_related(
            'organization',
            'project_type',
            'visibility',
            'created_by',
            'updated_by',
            'deleted_by'
        )


class ProjectMetadataInline(TabularInline):
    model = ProjectMetadata
    extra = 1  # Number of empty rows to display for quick addition
    fields = ('key', 'value')
    readonly_fields = ('created_at',)


class ProjectImageInline(TabularInline):
    model = ProjectImage
    extra = 1
    fields = ('image', 'annotated', 'added_at')
    readonly_fields = ('added_at',)


@admin.register(ProjectMetadata)
class ProjectMetadataAdmin(ModelAdmin):
    list_display = ('project', 'key', 'value', 'created_at')
    search_fields = ('project__name', 'key', 'value')
    list_filter = ('created_at',)
    ordering = ('-created_at',)


@admin.register(ImageMode)
class ImageModeAdmin(ModelAdmin):
    list_display = ('mode', 'description', 'created_at')
    search_fields = ('mode', 'description')
    list_filter = ('created_at',)

@admin.register(ProjectImage)
class ProjectImageAdmin(ModelAdmin):
    list_display = ('project', 'image', "status", 'annotated', 'reviewed', 'feedback_provided', 'added_at')
    search_fields = ('project__name', 'image__image_name')
    list_filter = ('annotated', 'added_at', 'project', 'annotated', 'reviewed', 'mode', "status", "feedback_provided", "marked_as_null", "is_active")
    ordering = ('-added_at',)

    actions = ['mark_reviewed_as_annotated']

    @admin.action(description="Mark reviewed=True and annotated=False as annotated=True")
    def mark_reviewed_as_annotated(self, request, queryset):
        """Bulk update images: set annotated=True where reviewed=True and annotated=False"""
        updated_count = queryset.filter(reviewed=True, annotated=False).update(annotated=True)
        self.message_user(
            request,
            f"{updated_count} images were successfully updated to annotated=True."
        )
        
@admin.register(Version)
class VersionAdmin(ModelAdmin):
    list_display = ('id', 'project', 'version_name', 'version_number', "version_file", 'created_at')
    list_filter = ('project', 'created_at')
    ordering = ('project', 'version_number')
    search_fields = ('project__name', 'version_name')

@admin.register(VersionImage)
class VersionImageAdmin(ModelAdmin):
    list_display = ('version', 'project_image', 'added_at')
    list_filter = ('version', 'version__project')