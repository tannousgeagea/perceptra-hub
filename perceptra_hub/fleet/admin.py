from django.contrib import admin
from unfold.admin import ModelAdmin, TabularInline

from .models import Camera, Device, DeviceModel, Rollout, RolloutTarget, Site


class CameraInline(TabularInline):
    model = Camera
    extra = 0


class DeviceModelInline(TabularInline):
    model = DeviceModel
    extra = 0
    readonly_fields = ('model', 'desired_version', 'reported_version', 'reported_status', 'reported_error', 'reported_at')
    can_delete = False


@admin.register(Site)
class SiteAdmin(ModelAdmin):
    list_display = ('name', 'organization', 'location', 'created_at')
    search_fields = ('name', 'location')
    list_filter = ('organization',)


@admin.register(Device)
class DeviceAdmin(ModelAdmin):
    list_display = ('name', 'organization', 'site', 'agent_version', 'last_seen_at', 'enrolled_at', 'revoked_at')
    search_fields = ('name',)
    list_filter = ('organization', 'site')
    # Credentials are only ever set through enrollment.
    exclude = ('key_hash', 'enrollment_code_hash')
    readonly_fields = ('device_id', 'key_prefix', 'enrolled_at', 'last_seen_at', 'last_ip', 'agent_version',
                       'protocol', 'hardware', 'accepts', 'health')
    inlines = [CameraInline, DeviceModelInline]


class RolloutTargetInline(TabularInline):
    model = RolloutTarget
    extra = 0
    readonly_fields = ('device', 'phase', 'state', 'error', 'assigned_at', 'finished_at')
    can_delete = False


@admin.register(Rollout)
class RolloutAdmin(ModelAdmin):
    list_display = ('model', 'version', 'strategy', 'status', 'created_by', 'created_at')
    list_filter = ('status', 'strategy')
    readonly_fields = ('rollout_id',)
    inlines = [RolloutTargetInline]
