
from django.contrib import admin
from unfold.admin import ModelAdmin
from django.utils.html import format_html
from unfold.decorators import display

from .models import (
    ModelTask,
    ModelFramework,
    ModelTag,
    Model,
    ModelVersion,
)

# Register your models here.

@admin.register(ModelTask)
class ModelTaskAdmin(ModelAdmin):
    list_display = ("name", "description", "created_at")
    search_fields = ("name", "description")
    ordering = ("name",)

@admin.register(ModelFramework)
class ModelFrameworkAdmin(ModelAdmin):
    list_display = ("name", "description", "created_at")
    search_fields = ("name", "description")
    ordering = ("name",)

@admin.register(ModelTag)
class ModelTagAdmin(ModelAdmin):
    list_display = ("name", "organization", "color_preview", "created_at")
    list_filter = ("organization",)
    search_fields = ("name", "description")
    readonly_fields = ("created_at",)

    @display(description="Color")
    def color_badge(self, obj):
        return format_html(
            '<span style="'
            'display:inline-block;'
            'width:18px;'
            'height:18px;'
            'border-radius:4px;'
            'background:{};'
            'border:1px solid #ccc;'
            '"></span>',
            obj.color,
        )
    
    @display(description="Color")
    def color_preview(self, obj):
        """Show color preview with hex code."""
        if obj.color:
            return format_html(
                '<div style="display: inline-block; width: 20px; height: 20px; '
                'background-color: {}; border: 1px solid #ccc; margin-right: 5px; '
                'vertical-align: middle;"></div>{}',
                obj.color,
                obj.color
            )
        return '-'

class ModelVersionInline(admin.TabularInline):
    model = ModelVersion
    extra = 0
    fields = (
        "version_number",
        "version_name",
        "status",
        "deployment_status",
        "created_at",
    )
    readonly_fields = fields
    ordering = ("-version_number",)
    show_change_link = True

@admin.register(Model)
class MLModelAdmin(ModelAdmin):
    inlines = [ModelVersionInline]

    list_display = (
        "name",
        "organization",
        "project",
        "task",
        "framework",
        "tag_list",
        "created_at",
        "is_deleted",
    )

    list_filter = (
        "organization",
        "project",
        "task",
        "framework",
        "is_deleted",
    )

    search_fields = ("name", "model_id", "description")
    readonly_fields = ("model_id", "created_at", "updated_at")

    filter_horizontal = ("tags",)

    fieldsets = (
        ("Basic Info", {
            "fields": ("model_id", "name", "description")
        }),
        ("Ownership", {
            "fields": ("organization", "project", "created_by")
        }),
        ("ML Configuration", {
            "fields": ("task", "framework", "default_config")
        }),
        ("Tags", {
            "fields": ("tags",)
        }),
        ("Audit", {
            "fields": (
                "created_at",
                "updated_at",
                "is_deleted",
                "deleted_at",
                "deleted_by",
            )
        }),
    )

    @display(description="Tags")
    def tag_list(self, obj):
        if not obj.tags.exists():
            return "-"

        return format_html(
            " ".join(
                '<span style="'
                'display:inline-flex;'
                'align-items:center;'
                'padding:2px 8px;'
                'margin-right:4px;'
                'border-radius:999px;'
                'font-size:12px;'
                'font-weight:500;'
                'line-height:1.4;'
                'background:{color}20;'
                'color:{color};'
                'border:1px solid {color}40;'
                '">'
                '{name}'
                '</span>'.format(
                    color=tag.color,
                    name=tag.name,
                )
                for tag in obj.tags.all()
            )
        )

    
@admin.register(ModelVersion)
class ModelVersionAdmin(ModelAdmin):
    list_display = (
        "model",
        "version_number",
        "version_name",
        "status_badge",
        "deployment_badge",
        "dataset_version",
        "created_at",
    )

    list_filter = (
        "model",
        "status",
        "deployment_status",
        "is_deleted",
    )

    search_fields = (
        "version_id",
        "version_name",
        "model__name",
    )

    readonly_fields = (
        "version_id",
        "created_at",
        "updated_at",
        "deployed_at",
    )

    fieldsets = (
        ("Version Info", {
            "fields": (
                "model",
                "version_id",
                "version_number",
                "version_name",
                "parent_version",
            )
        }),
        ("Dataset", {
            "fields": ("dataset_version",)
        }),
        ("Storage", {
            "fields": (
                "storage_profile",
                "checkpoint_key",
                "onnx_model_key",
                "training_logs_key",
            )
        }),
        ("Training", {
            "fields": ("config", "metrics", "status", "error_message")
        }),
        ("Deployment", {
            "fields": (
                "deployment_status",
                "deployment_config",
                "deployed_at",
                "deployed_by",
            )
        }),
        ("Audit", {
            "fields": (
                "created_by",
                "created_at",
                "updated_at",
                "is_deleted",
                "deleted_at",
            )
        }),
    )

    @display(description="Status")
    def status_badge(self, obj):
        color = {
            "draft": "gray",
            "queued": "blue",
            "training": "orange",
            "trained": "green",
            "failed": "red",
            "cancelled": "darkred",
        }.get(obj.status, "gray")

        return f"<span style='color:{color};font-weight:600'>{obj.status}</span>"

    @display(description="Deployment")
    def deployment_badge(self, obj):
        color = {
            "none": "gray",
            "staging": "orange",
            "production": "green",
            "retired": "darkred",
        }.get(obj.deployment_status, "gray")

        return f"<span style='color:{color};font-weight:600'>{obj.deployment_status}</span>"

    
    