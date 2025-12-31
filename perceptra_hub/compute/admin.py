"""
Django Admin for Compute models.
Location: compute/admin.py
"""
from django.contrib import admin
from unfold.admin import ModelAdmin
from django.utils.html import format_html
from .models import ComputeProvider, ComputeProfile, ComputeFallback, TrainingJob, Agent


@admin.register(ComputeProvider)
class ComputeProviderAdmin(ModelAdmin):
    list_display = [
        'name', 
        'provider_type', 
        'requires_user_credentials',
        'is_active',
        'instance_count',
        'created_at'
    ]
    list_filter = ['provider_type', 'is_active', 'requires_user_credentials']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'provider_type', 'description', 'is_active')
        }),
        ('Configuration', {
            'fields': ('system_config', 'available_instances', 'requires_user_credentials')
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def instance_count(self, obj):
        """Show number of available instances"""
        return len(obj.available_instances) if obj.available_instances else 0
    instance_count.short_description = 'Instances'
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.prefetch_related('compute_profiles')


@admin.register(ComputeProfile)
class ComputeProfileAdmin(ModelAdmin):
    list_display = [
        'name',
        'organization',
        'provider',
        'strategy',
        'is_active',
        'is_default',
        'max_concurrent_jobs',
        'active_jobs_count',
        'created_at'
    ]
    list_filter = [
        'strategy',
        'is_active',
        'is_default',
        'provider__provider_type'
    ]
    search_fields = ['name', 'organization__name', 'provider__name']
    readonly_fields = ['profile_id', 'created_at', 'updated_at', 'active_jobs_count']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('profile_id', 'name', 'organization', 'provider')
        }),
        ('Configuration', {
            'fields': (
                'default_instance_type',
                'strategy',
                'max_concurrent_jobs',
                'max_cost_per_hour',
                'max_training_hours'
            )
        }),
        ('Status', {
            'fields': ('is_active', 'is_default')
        }),
        ('Credentials', {
            'fields': ('user_credentials',),
            'classes': ('collapse',),
            'description': 'Encrypted user credentials (view only in production)'
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at', 'active_jobs_count'),
            'classes': ('collapse',)
        }),
    )
    
    def active_jobs_count(self, obj):
        """Count active training jobs"""
        from compute.models import TrainingJob
        count = TrainingJob.objects.filter(
            compute_profile=obj,
            training_session__status__in=['queued', 'running', 'initializing']
        ).count()
        if count > 0:
            return format_html('<span style="color: green; font-weight: bold;">{}</span>', count)
        return count
    active_jobs_count.short_description = 'Active Jobs'
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related('organization', 'provider', 'created_by')


@admin.register(ComputeFallback)
class ComputeFallbackAdmin(ModelAdmin):
    list_display = ['profile', 'provider', 'priority']
    list_filter = ['profile__organization']
    search_fields = ['profile__name', 'provider__name']
    ordering = ['profile', 'priority']
    
    fieldsets = (
        (None, {
            'fields': ('profile', 'provider', 'priority')
        }),
    )


@admin.register(TrainingJob)
class TrainingJobAdmin(ModelAdmin):
    list_display = [
        'job_id_short',
        'model_version_link',
        'actual_provider',
        'instance_type',
        'status',
        'estimated_cost',
        'actual_cost',
        'started_at',
        'duration'
    ]
    list_filter = [
        'actual_provider__provider_type',
        'training_session__status',
        'started_at'
    ]
    search_fields = [
        'job_id',
        'external_job_id',
        'training_session__model_version__model__name'
    ]
    readonly_fields = [
        'job_id',
        'training_session',
        'compute_profile',
        'actual_provider',
        'instance_type',
        'external_job_id',
        'estimated_cost',
        'actual_cost',
        'gpu_hours',
        'started_at',
        'completed_at',
        'duration'
    ]
    date_hierarchy = 'started_at'
    
    fieldsets = (
        ('Job Information', {
            'fields': ('job_id', 'external_job_id', 'training_session')
        }),
        ('Compute Configuration', {
            'fields': ('compute_profile', 'actual_provider', 'instance_type')
        }),
        ('Cost Tracking', {
            'fields': ('estimated_cost', 'actual_cost', 'gpu_hours')
        }),
        ('Timing', {
            'fields': ('started_at', 'completed_at', 'duration')
        }),
    )
    
    def job_id_short(self, obj):
        """Show shortened job ID"""
        return obj.job_id[:12] + '...' if len(obj.job_id) > 12 else obj.job_id
    job_id_short.short_description = 'Job ID'
    
    def model_version_link(self, obj):
        """Link to model version"""
        if obj.training_session and obj.training_session.model_version:
            mv = obj.training_session.model_version
            return format_html(
                '{} v{}',
                mv.model.name,
                mv.version_number
            )
        return '-'
    model_version_link.short_description = 'Model Version'
    
    def status(self, obj):
        """Show colorized status"""
        if obj.training_session:
            status = obj.training_session.status
            colors = {
                'completed': 'green',
                'running': 'blue',
                'failed': 'red',
                'queued': 'orange',
                'cancelled': 'gray'
            }
            color = colors.get(status, 'black')
            return format_html(
                '<span style="color: {}; font-weight: bold;">{}</span>',
                color,
                status.upper()
            )
        return '-'
    status.short_description = 'Status'
    
    def duration(self, obj):
        """Calculate job duration"""
        if obj.started_at and obj.completed_at:
            delta = obj.completed_at - obj.started_at
            hours = delta.total_seconds() / 3600
            return f"{hours:.2f}h"
        return '-'
    duration.short_description = 'Duration'
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related(
            'training_session__model_version__model',
            'compute_profile',
            'actual_provider'
        )


@admin.register(Agent)
class AgentAdmin(ModelAdmin):
    list_display = [
        'agent_id_short',
        'name',
        'organization',
        'status_indicator',
        'gpu_count',
        'last_heartbeat',
        'uptime'
    ]
    list_filter = ['status', 'organization', 'created_at']
    search_fields = ['agent_id', 'name', 'organization__name']
    readonly_fields = [
        'agent_id',
        'gpu_info',
        'last_heartbeat',
        'created_at',
        'uptime'
    ]
    
    fieldsets = (
        ('Agent Information', {
            'fields': ('agent_id', 'name', 'organization', 'status')
        }),
        ('Hardware', {
            'fields': ('gpu_info',)
        }),
        ('Status', {
            'fields': ('last_heartbeat', 'created_at', 'uptime')
        }),
    )
    
    def agent_id_short(self, obj):
        """Show shortened agent ID"""
        return obj.agent_id[:12]
    agent_id_short.short_description = 'Agent ID'
    
    def status_indicator(self, obj):
        """Show colorized status"""
        colors = {
            'ready': 'green',
            'busy': 'blue',
            'offline': 'red'
        }
        color = colors.get(obj.status, 'gray')
        return format_html(
            '<span style="color: {}; font-weight: bold;">●</span> {}',
            color,
            obj.status.upper()
        )
    status_indicator.short_description = 'Status'
    
    def gpu_count(self, obj):
        """Show GPU count"""
        if obj.gpu_info:
            return len(obj.gpu_info)
        return 0
    gpu_count.short_description = 'GPUs'
    
    def uptime(self, obj):
        """Calculate uptime"""
        from django.utils import timezone
        if obj.last_heartbeat:
            delta = timezone.now() - obj.created_at
            days = delta.days
            hours = delta.seconds // 3600
            return f"{days}d {hours}h"
        return '-'
    uptime.short_description = 'Uptime'
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related('organization')