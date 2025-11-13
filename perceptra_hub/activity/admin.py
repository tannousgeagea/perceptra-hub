# apps/activity/admin.py
from django.contrib import admin
from unfold.admin import ModelAdmin, TabularInline
from django.utils.html import format_html
from django.db.models import Count, Sum, Avg
from django.utils import timezone
from datetime import timedelta
import json

from .models import (
    ActivityEvent,
    UserActivityMetrics,
    ProjectActivityMetrics,
    UserSessionActivity
)


# ============================================================================
# Custom Filters
# ============================================================================

class TimeRangeFilter(admin.SimpleListFilter):
    """Filter by time ranges."""
    title = 'time range'
    parameter_name = 'time_range'

    def lookups(self, request, model_admin):
        return (
            ('1h', 'Last Hour'),
            ('24h', 'Last 24 Hours'),
            ('7d', 'Last 7 Days'),
            ('30d', 'Last 30 Days'),
            ('90d', 'Last 90 Days'),
        )

    def queryset(self, request, queryset):
        if self.value() == '1h':
            time_threshold = timezone.now() - timedelta(hours=1)
        elif self.value() == '24h':
            time_threshold = timezone.now() - timedelta(days=1)
        elif self.value() == '7d':
            time_threshold = timezone.now() - timedelta(days=7)
        elif self.value() == '30d':
            time_threshold = timezone.now() - timedelta(days=30)
        elif self.value() == '90d':
            time_threshold = timezone.now() - timedelta(days=90)
        else:
            return queryset
        
        return queryset.filter(timestamp__gte=time_threshold)


class EventCategoryFilter(admin.SimpleListFilter):
    """Filter by event category."""
    title = 'event category'
    parameter_name = 'category'

    def lookups(self, request, model_admin):
        return (
            ('image', 'Image Events'),
            ('annotation', 'Annotation Events'),
            ('review', 'Review Events'),
            ('job', 'Job Events'),
            ('prediction', 'AI/Prediction Events'),
            ('dataset', 'Dataset Events'),
        )

    def queryset(self, request, queryset):
        if self.value() == 'image':
            return queryset.filter(event_type__startswith='image_')
        elif self.value() == 'annotation':
            return queryset.filter(event_type__startswith='annotation_')
        elif self.value() == 'review':
            return queryset.filter(event_type__in=[
                'annotation_review', 'image_review', 'image_finalize'
            ])
        elif self.value() == 'job':
            return queryset.filter(event_type__startswith='job_')
        elif self.value() == 'prediction':
            return queryset.filter(event_type__startswith='prediction_')
        elif self.value() == 'dataset':
            return queryset.filter(event_type__startswith='dataset_')
        return queryset


class ActiveSessionFilter(admin.SimpleListFilter):
    """Filter active vs inactive sessions."""
    title = 'session status'
    parameter_name = 'active'

    def lookups(self, request, model_admin):
        return (
            ('active', 'Active (last 30 min)'),
            ('recent', 'Recent (last 24h)'),
            ('stale', 'Stale (>24h)'),
        )

    def queryset(self, request, queryset):
        now = timezone.now()
        if self.value() == 'active':
            return queryset.filter(
                last_activity__gte=now - timedelta(minutes=30),
                is_active=True
            )
        elif self.value() == 'recent':
            return queryset.filter(
                last_activity__gte=now - timedelta(days=1)
            )
        elif self.value() == 'stale':
            return queryset.filter(
                last_activity__lt=now - timedelta(days=1)
            )
        return queryset


# ============================================================================
# Activity Event Admin
# ============================================================================

@admin.register(ActivityEvent)
class ActivityEventAdmin(ModelAdmin):
    """
    Read-only admin for activity events with powerful filtering and search.
    """
    
    list_display = [
        'event_id_short',
        'timestamp',
        'event_type_badge',
        'user_link',
        'organization',
        'project_link',
        'source',
        'duration_display',
        'session_id_short',
    ]
    
    list_filter = [
        TimeRangeFilter,
        EventCategoryFilter,
        'event_type',
        'source',
        'organization',
    ]
    
    search_fields = [
        'event_id',
        'user__username',
        'user__email',
        'project__name',
        'organization__name',
        'session_id',
    ]
    
    readonly_fields = [
        'event_id',
        'organization',
        'event_type',
        'user',
        'project',
        'timestamp',
        'metadata_pretty',
        'duration_ms',
        'session_id',
        'source',
    ]
    
    date_hierarchy = 'timestamp'
    
    ordering = ['-timestamp']
    
    list_per_page = 50
    
    # Disable add/delete/change permissions (append-only log)
    def has_add_permission(self, request):
        return False
    
    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser
    
    def has_change_permission(self, request, obj=None):
        return False
    
    # Custom display methods
    def event_id_short(self, obj):
        """Display short event ID with copy tooltip."""
        return format_html(
            '<span title="{}" style="font-family: monospace;">{}</span>',
            obj.event_id,
            str(obj.event_id)[:8]
        )
    event_id_short.short_description = 'Event ID'
    
    def event_type_badge(self, obj):
        """Display event type as colored badge."""
        colors = {
            'image': '#3498db',      # Blue
            'annotation': '#2ecc71', # Green
            'review': '#f39c12',     # Orange
            'job': '#9b59b6',        # Purple
            'prediction': '#1abc9c', # Teal
            'dataset': '#e74c3c',    # Red
        }
        
        category = obj.event_type.split('_')[0]
        color = colors.get(category, '#95a5a6')
        
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; '
            'border-radius: 3px; font-size: 11px; font-weight: bold;">{}</span>',
            color,
            obj.get_event_type_display()
        )
    event_type_badge.short_description = 'Event Type'
    
    def user_link(self, obj):
        """Link to user admin."""
        if obj.user:
            return format_html(
                '<a href="/admin/auth/user/{}/change/">{}</a>',
                obj.user.id,
                obj.user.username
            )
        return '-'
    user_link.short_description = 'User'
    
    def project_link(self, obj):
        """Link to project admin."""
        if obj.project:
            return format_html(
                '<a href="/admin/projects/project/{}/change/">{}</a>',
                obj.project.id,
                obj.project.name
            )
        return '-'
    project_link.short_description = 'Project'
    
    def duration_display(self, obj):
        """Format duration in human-readable format."""
        if obj.duration_ms is None:
            return '-'
        
        if obj.duration_ms < 1000:
            return f"{obj.duration_ms}ms"
        elif obj.duration_ms < 60000:
            return f"{obj.duration_ms / 1000}s"
        else:
            minutes = obj.duration_ms / 60000
            return f"{minutes}min"
    duration_display.short_description = 'Duration'
    
    def session_id_short(self, obj):
        """Display short session ID."""
        if obj.session_id:
            return format_html(
                '<span title="{}" style="font-family: monospace;">{}</span>',
                obj.session_id,
                str(obj.session_id)[:8]
            )
        return '-'
    session_id_short.short_description = 'Session'
    
    def metadata_pretty(self, obj):
        """Display formatted JSON metadata."""
        if not obj.metadata:
            return '-'
        
        try:
            formatted = json.dumps(obj.metadata, indent=2)
            return format_html(
                '<pre style="background: #f5f5f5; padding: 10px; '
                'border-radius: 4px; max-width: 600px; overflow-x: auto;">{}</pre>',
                formatted
            )
        except:
            return str(obj.metadata)
    metadata_pretty.short_description = 'Metadata'
    
    # Custom actions
    actions = ['export_as_csv']
    
    def export_as_csv(self, request, queryset):
        """Export selected events as CSV."""
        import csv
        from django.http import HttpResponse
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="activity_events.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'Event ID', 'Timestamp', 'Event Type', 'User', 'Organization',
            'Project', 'Source', 'Duration (ms)', 'Metadata'
        ])
        
        for event in queryset:
            writer.writerow([
                str(event.event_id),
                event.timestamp.isoformat(),
                event.event_type,
                event.user.username if event.user else '',
                event.organization.name,
                event.project.name if event.project else '',
                event.source,
                event.duration_ms or '',
                json.dumps(event.metadata)
            ])
        
        return response
    export_as_csv.short_description = 'Export selected events as CSV'


# ============================================================================
# User Activity Metrics Admin
# ============================================================================

@admin.register(UserActivityMetrics)
class UserActivityMetricsAdmin(ModelAdmin):
    """
    Admin for user activity metrics with aggregated statistics.
    """
    
    list_display = [
        'user',
        'organization',
        'project',
        'period_display',
        'granularity',
        'annotations_created',
        'images_reviewed',
        'ai_predictions_accepted',
        'productivity_score',
        'last_activity',
    ]
    
    list_filter = [
        'granularity',
        'organization',
        'project',
        'period_start',
    ]
    
    search_fields = [
        'user__username',
        'user__email',
        'organization__name',
        'project__name',
    ]
    
    readonly_fields = [
        'user',
        'organization',
        'project',
        'period_start',
        'period_end',
        'granularity',
        'last_activity',
        'updated_at',
        'metrics_summary',
    ]
    
    date_hierarchy = 'period_start'
    
    ordering = ['-period_start', '-annotations_created']
    
    list_per_page = 50
    
    # Disable add (metrics are auto-generated)
    def has_add_permission(self, request):
        return False
    
    def period_display(self, obj):
        """Display period in readable format."""
        if obj.granularity == 'hour':
            return obj.period_start.strftime('%Y-%m-%d %H:00')
        elif obj.granularity == 'day':
            return obj.period_start.strftime('%Y-%m-%d')
        elif obj.granularity == 'week':
            return f"Week of {obj.period_start.strftime('%Y-%m-%d')}"
        else:
            return obj.period_start.strftime('%Y-%m')
    period_display.short_description = 'Period'
    
    def productivity_score(self, obj):
        """Calculate simple productivity score."""
        score = (
            obj.annotations_created * 2 +
            obj.images_reviewed +
            obj.ai_predictions_accepted * 0.5
        )
        
        color = '#2ecc71' if score > 100 else '#f39c12' if score > 50 else '#95a5a6'
        
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            score
        )
    productivity_score.short_description = 'Score'
    
    def metrics_summary(self, obj):
        """Display comprehensive metrics summary."""
        html = f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 800px;">
            <div>
                <h3>📸 Image Activity</h3>
                <ul>
                    <li>Uploaded: <strong>{obj.images_uploaded}</strong></li>
                    <li>Added to Project: <strong>{obj.images_added_to_project}</strong></li>
                    <li>Total Size: <strong>{obj.total_upload_size_mb} MB</strong></li>
                </ul>
                
                <h3>✏️ Annotation Activity</h3>
                <ul>
                    <li>Created: <strong>{obj.annotations_created}</strong></li>
                    <li>Updated: <strong>{obj.annotations_updated}</strong></li>
                    <li>Deleted: <strong>{obj.annotations_deleted}</strong></li>
                    <li>Manual: <strong>{obj.manual_annotations}</strong></li>
                </ul>
            </div>
            
            <div>
                <h3>🤖 AI Predictions</h3>
                <ul>
                    <li>Accepted: <strong>{obj.ai_predictions_accepted}</strong></li>
                    <li>Edited: <strong>{obj.ai_predictions_edited}</strong></li>
                    <li>Rejected: <strong>{obj.ai_predictions_rejected}</strong></li>
                </ul>
                
                <h3>✅ Review Activity</h3>
                <ul>
                    <li>Images Reviewed: <strong>{obj.images_reviewed}</strong></li>
                    <li>Approved: <strong>{obj.images_approved}</strong></li>
                    <li>Rejected: <strong>{obj.images_rejected}</strong></li>
                    <li>Finalized: <strong>{obj.images_finalized}</strong></li>
                    <li>Annotations Reviewed: <strong>{obj.annotations_reviewed}</strong></li>
                </ul>
                
                <h3>📊 Quality Metrics</h3>
                <ul>
                    <li>Avg Annotation Time: <strong>{obj.avg_annotation_time_seconds or 'N/A'} sec</strong></li>
                    <li>Avg Edit Magnitude: <strong>{obj.avg_edit_magnitude or 'N/A'}</strong></li>
                </ul>
                
                <h3>💼 Job Completion</h3>
                <ul>
                    <li>Jobs Completed: <strong>{obj.jobs_completed}</strong></li>
                </ul>
            </div>
        </div>
        """
        return format_html(html)
    metrics_summary.short_description = 'Detailed Metrics'
    
    # Custom actions
    actions = ['recalculate_metrics']
    
    def recalculate_metrics(self, request, queryset):
        """Trigger recalculation of selected metrics (placeholder)."""
        count = queryset.count()
        self.message_user(
            request,
            f"Recalculation queued for {count} metric records."
        )
    recalculate_metrics.short_description = 'Recalculate selected metrics'


# ============================================================================
# Project Activity Metrics Admin
# ============================================================================

@admin.register(ProjectActivityMetrics)
class ProjectActivityMetricsAdmin(ModelAdmin):
    """
    Admin for project-level activity metrics.
    """
    
    list_display = [
        'project',
        'period_display',
        'granularity',
        'progress_bar',
        'total_annotations',
        'active_users',
        'velocity_display',
        'updated_at',
    ]
    
    list_filter = [
        'granularity',
        'organization',
        'period_start',
    ]
    
    search_fields = [
        'project__name',
        'organization__name',
    ]
    
    readonly_fields = [
        'project',
        'organization',
        'period_start',
        'period_end',
        'granularity',
        'updated_at',
        'project_overview',
        'progress_breakdown',
        'quality_metrics_display',
    ]
    
    date_hierarchy = 'period_start'
    
    ordering = ['-period_start']
    
    list_per_page = 50
    
    def has_add_permission(self, request):
        return False
    
    def period_display(self, obj):
        """Display period."""
        if obj.granularity == 'day':
            return obj.period_start.strftime('%Y-%m-%d')
        elif obj.granularity == 'week':
            return f"Week of {obj.period_start.strftime('%Y-%m-%d')}"
        else:
            return obj.period_start.strftime('%Y-%m')
    period_display.short_description = 'Period'
    
    def progress_bar(self, obj):
        """Visual progress bar."""
        if obj.total_images == 0:
            return '-'
        
        finalized_pct = (obj.images_finalized / obj.total_images) * 100
        reviewed_pct = (obj.images_reviewed / obj.total_images) * 100
        annotated_pct = (obj.images_annotated / obj.total_images) * 100
        
        return format_html(
            '<div style="width: 200px; background: #ecf0f1; border-radius: 4px; overflow: hidden;">'
            '<div style="width: {}%; background: #2ecc71; height: 20px; float: left;"></div>'
            '<div style="width: {}%; background: #3498db; height: 20px; float: left;"></div>'
            '<div style="width: {}%; background: #f39c12; height: 20px; float: left;"></div>'
            '</div>'
            '<div style="font-size: 11px; margin-top: 2px;">'
            '✓ {}% | ⚡ {}% | ✏️ {}%'
            '</div>',
            finalized_pct, reviewed_pct - finalized_pct, annotated_pct - reviewed_pct,
            finalized_pct, reviewed_pct, annotated_pct
        )
    progress_bar.short_description = 'Progress'
    
    def velocity_display(self, obj):
        """Display annotation velocity."""
        if obj.annotations_per_hour:
            return f"{obj.annotations_per_hour}/hr"
        return '-'
    velocity_display.short_description = 'Velocity'
    
    def project_overview(self, obj):
        """Display project overview."""
        html = f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
            <div style="background: #ecf0f1; padding: 15px; border-radius: 5px;">
                <h3>📊 Image Status</h3>
                <ul>
                    <li>Total: <strong>{obj.total_images}</strong></li>
                    <li>Unannotated: <strong>{obj.images_unannotated}</strong></li>
                    <li>Annotated: <strong>{obj.images_annotated}</strong></li>
                    <li>Reviewed: <strong>{obj.images_reviewed}</strong></li>
                    <li>Finalized: <strong>{obj.images_finalized}</strong></li>
                </ul>
            </div>
            
            <div style="background: #e8f5e9; padding: 15px; border-radius: 5px;">
                <h3>✏️ Annotations</h3>
                <ul>
                    <li>Total: <strong>{obj.total_annotations}</strong></li>
                    <li>Manual: <strong>{obj.manual_annotations}</strong></li>
                    <li>AI Predictions: <strong>{obj.ai_predictions}</strong></li>
                    <li>Avg per Image: <strong>{obj.avg_annotations_per_image or 'N/A'}</strong></li>
                </ul>
            </div>
            
            <div style="background: #e3f2fd; padding: 15px; border-radius: 5px;">
                <h3>👥 Contributors</h3>
                <ul>
                    <li>Active Users: <strong>{obj.active_users}</strong></li>
                    <li>Velocity: <strong>{obj.annotations_per_hour or 'N/A'} ann/hr</strong></li>
                </ul>
            </div>
        </div>
        """
        return format_html(html)
    project_overview.short_description = 'Overview'
    
    def progress_breakdown(self, obj):
        """Visual progress breakdown."""
        if obj.total_images == 0:
            return format_html('<p>No images in project</p>')
        
        unannotated_pct = (obj.images_unannotated / obj.total_images) * 100
        annotated_pct = (obj.images_annotated / obj.total_images) * 100
        reviewed_pct = (obj.images_reviewed / obj.total_images) * 100
        finalized_pct = (obj.images_finalized / obj.total_images) * 100
        
        html = f"""
        <div style="max-width: 600px;">
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>❌ Unannotated</span>
                    <span><strong>{obj.images_unannotated}</strong> ({unannotated_pct}%)</span>
                </div>
                <div style="width: 100%; background: #ecf0f1; border-radius: 3px;">
                    <div style="width: {unannotated_pct}%; background: #e74c3c; height: 15px; border-radius: 3px;"></div>
                </div>
            </div>
            
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>✏️ Annotated</span>
                    <span><strong>{obj.images_annotated}</strong> ({annotated_pct}%)</span>
                </div>
                <div style="width: 100%; background: #ecf0f1; border-radius: 3px;">
                    <div style="width: {annotated_pct}%; background: #f39c12; height: 15px; border-radius: 3px;"></div>
                </div>
            </div>
            
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>⚡ Reviewed</span>
                    <span><strong>{obj.images_reviewed}</strong> ({reviewed_pct}%)</span>
                </div>
                <div style="width: 100%; background: #ecf0f1; border-radius: 3px;">
                    <div style="width: {reviewed_pct}%; background: #3498db; height: 15px; border-radius: 3px;"></div>
                </div>
            </div>
            
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>✅ Finalized</span>
                    <span><strong>{obj.images_finalized}</strong> ({finalized_pct}%)</span>
                </div>
                <div style="width: 100%; background: #ecf0f1; border-radius: 3px;">
                    <div style="width: {finalized_pct}%; background: #2ecc71; height: 15px; border-radius: 3px;"></div>
                </div>
            </div>
        </div>
        """
        return format_html(html)
    progress_breakdown.short_description = 'Progress Breakdown'
    
    def quality_metrics_display(self, obj):
        """Display quality metrics."""
        ai_acceptance_rate = 'N/A'
        ai_edit_rate = 'N/A'
        
        if obj.ai_predictions > 0:
            accepted = obj.ai_predictions - obj.edited_predictions - obj.rejected_predictions
            ai_acceptance_rate = f"{(accepted / obj.ai_predictions) * 100}%"
            ai_edit_rate = f"{(obj.edited_predictions / obj.ai_predictions) * 100}%"
        
        html = f"""
        <div style="background: #fff3cd; padding: 15px; border-radius: 5px; max-width: 500px;">
            <h3>🎯 Quality Metrics</h3>
            <ul>
                <li>Untouched Predictions: <strong>{obj.untouched_predictions}</strong></li>
                <li>Edited Predictions: <strong>{obj.edited_predictions}</strong></li>
                <li>Rejected Predictions: <strong>{obj.rejected_predictions}</strong></li>
                <li>AI Acceptance Rate: <strong>{ai_acceptance_rate}</strong></li>
                <li>AI Edit Rate: <strong>{ai_edit_rate}</strong></li>
            </ul>
        </div>
        """
        return format_html(html)
    quality_metrics_display.short_description = 'Quality Metrics'


# ============================================================================
# User Session Activity Admin
# ============================================================================

@admin.register(UserSessionActivity)
class UserSessionActivityAdmin(ModelAdmin):
    """
    Admin for real-time user session tracking.
    """
    
    list_display = [
        'session_id_short',
        'user',
        'organization',
        'project',
        'started_at',
        'last_activity',
        'duration_display',
        'actions_count',
        'annotations_created',
        'images_processed',
        'status_badge',
    ]
    
    list_filter = [
        ActiveSessionFilter,
        'is_active',
        'organization',
        'project',
        'started_at',
    ]
    
    search_fields = [
        'session_id',
        'user__username',
        'user__email',
        'organization__name',
        'project__name',
    ]
    
    readonly_fields = [
        'session_id',
        'user',
        'organization',
        'project',
        'started_at',
        'last_activity',
        'actions_count',
        'annotations_created',
        'images_processed',
        'is_active',
        'session_duration',
    ]
    
    date_hierarchy = 'started_at'
    
    ordering = ['-last_activity']
    
    list_per_page = 50
    
    def has_add_permission(self, request):
        return False
    
    def session_id_short(self, obj):
        """Display short session ID."""
        return format_html(
            '<span title="{}" style="font-family: monospace;">{}</span>',
            obj.session_id,
            str(obj.session_id)[:8]
        )
    session_id_short.short_description = 'Session'
    
    def duration_display(self, obj):
        """Display session duration."""
        duration = obj.last_activity - obj.started_at
        total_seconds = int(duration.total_seconds())
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m"
        else:
            return f"{total_seconds}s"
    duration_display.short_description = 'Duration'
    
    def status_badge(self, obj):
        """Display session status as badge."""
        now = timezone.now()
        time_since_activity = (now - obj.last_activity).total_seconds()
        
        if obj.is_active and time_since_activity < 1800:  # 30 minutes
            color = '#2ecc71'
            status = 'ACTIVE'
        elif time_since_activity < 86400:  # 24 hours
            color = '#f39c12'
            status = 'RECENT'
        else:
            color = '#95a5a6'
            status = 'STALE'
        
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; '
            'border-radius: 3px; font-size: 11px; font-weight: bold;">{}</span>',
            color,
            status
        )
    status_badge.short_description = 'Status'
    
    def session_duration(self, obj):
        """Detailed session duration."""
        duration = obj.last_activity - obj.started_at
        return str(duration).split('.')[0]  # Remove microseconds
    session_duration.short_description = 'Total Duration'
    
    # Custom actions
    actions = ['mark_inactive', 'cleanup_stale_sessions']
    
    def mark_inactive(self, request, queryset):
        """Mark selected sessions as inactive."""
        count = queryset.update(is_active=False)
        self.message_user(request, f"{count} sessions marked as inactive.")
    mark_inactive.short_description = 'Mark as inactive'
    
    def cleanup_stale_sessions(self, request, queryset):
        """Delete sessions older than 24 hours."""
        threshold = timezone.now() - timedelta(days=1)
        count = queryset.filter(last_activity__lt=threshold).delete()[0]
        self.message_user(request, f"{count} stale sessions deleted.")
    cleanup_stale_sessions.short_description = 'Delete stale sessions (>24h)'