"""
Django Admin for API Keys
"""
from django.contrib import admin
from unfold.admin import ModelAdmin
from django.utils.html import format_html
from .models import APIKey, APIKeyUsageLog, APIKeyRateLimit


@admin.register(APIKey)
class APIKeyAdmin(ModelAdmin):
    list_display = [
        'display_prefix',
        'name',
        'organization',
        'scope',
        'permissions',
        'is_active_badge',
        'usage_count',
        'last_used_at',
        'expires_at',
    ]

    list_filter = [
        'scope',
        'permissions',
        'is_active',
        'created_at',
        'expires_at',
    ]

    search_fields = [
        'name',
        'key_prefix',
        'organization__name',
        'owned_by__username',
        'created_by__username',
    ]

    readonly_fields = [
        'api_key_id',
        'key_prefix',
        'hashed_key',
        'usage_count',
        'last_used_at',
        'created_at',
        'created_by',
        'version',
        'rotated_from',
    ]

    fieldsets = [
        ('Key Information', {
            'fields': ('api_key_id', 'key_prefix', 'name', 'description', 'hashed_key'),
        }),
        ('Ownership', {
            'fields': ('organization', 'scope', 'owned_by', 'created_by'),
        }),
        ('Permissions & Scopes', {
            'fields': ('permissions', 'scopes', 'allowed_ips', 'is_active'),
        }),
        ('Rate Limiting', {
            'fields': ('rate_limit_per_minute', 'rate_limit_per_hour'),
        }),
        ('Usage', {
            'fields': ('usage_count', 'last_used_at', 'created_at', 'expires_at'),
        }),
        ('Rotation', {
            'fields': ('version', 'rotated_from'),
            'classes': ('collapse',),
        }),
    ]

    def display_prefix(self, obj):
        return f"{obj.key_prefix}..."
    display_prefix.short_description = "Key Prefix"

    def is_active_badge(self, obj):
        if obj.is_active:
            color = 'green'
            text = 'Active'
        else:
            color = 'red'
            text = 'Inactive'
        return format_html(
            '<span style="color: {};">{}</span>',
            color,
            text,
        )
    is_active_badge.short_description = "Status"

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return True


@admin.register(APIKeyUsageLog)
class APIKeyUsageLogAdmin(ModelAdmin):
    list_display = [
        'timestamp',
        'api_key_prefix',
        'method',
        'endpoint',
        'status_code_badge',
        'response_time_ms',
        'ip_address',
    ]

    list_filter = [
        'method',
        'status_code',
        'timestamp',
    ]

    search_fields = [
        'api_key__key_prefix',
        'endpoint',
        'ip_address',
    ]

    readonly_fields = [
        'api_key',
        'timestamp',
        'endpoint',
        'method',
        'status_code',
        'response_time_ms',
        'ip_address',
        'user_agent',
    ]

    def api_key_prefix(self, obj):
        return f"{obj.api_key.key_prefix}..."
    api_key_prefix.short_description = "API Key"

    def status_code_badge(self, obj):
        if 200 <= obj.status_code < 300:
            color = 'green'
        elif 400 <= obj.status_code < 500:
            color = 'orange'
        else:
            color = 'red'

        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            obj.status_code,
        )
    status_code_badge.short_description = "Status"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(APIKeyRateLimit)
class APIKeyRateLimitAdmin(ModelAdmin):
    list_display = [
        'api_key_prefix',
        'window_type',
        'window_start',
        'request_count',
        'limit_status',
    ]

    list_filter = [
        'window_type',
        'window_start',
    ]

    search_fields = [
        'api_key__key_prefix',
    ]

    readonly_fields = [
        'api_key',
        'window_start',
        'window_type',
        'request_count',
    ]

    def api_key_prefix(self, obj):
        return f"{obj.api_key.key_prefix}..."
    api_key_prefix.short_description = "API Key"

    def limit_status(self, obj):
        if obj.window_type == 'minute':
            limit = obj.api_key.rate_limit_per_minute
        else:
            limit = obj.api_key.rate_limit_per_hour

        if limit == 0:
            return "Unlimited"

        percentage = (obj.request_count / limit) * 100 if limit > 0 else 0

        if percentage < 70:
            color = 'green'
        elif percentage < 90:
            color = 'orange'
        else:
            color = 'red'

        return format_html(
            '<span style="color: {};">{}/{} ({}%)</span>',
            color,
            obj.request_count,
            limit,
            round(percentage, 1),
        )
    limit_status.short_description = "Usage"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
