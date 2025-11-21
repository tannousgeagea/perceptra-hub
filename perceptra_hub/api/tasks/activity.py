# apps/activity/tasks.py
from celery import shared_task
from django.db.models import Count, Avg, Sum, F, Q, Max, FloatField
from django.db.models.functions import Cast
from django.utils import timezone
from datetime import timedelta, datetime
from activity.models import (
    ActivityEvent,
    UserActivityMetrics,
    ProjectActivityMetrics,
    UserSessionActivity,
    ActivityEventType
)
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# INCREMENTAL UPDATES (Real-time)
# ============================================================================

@shared_task(bind=True, max_retries=3, name="activity:update_user_metrics_async", queue="activity")
def update_user_metrics_async(self, user_id, organization_id, event_type, project_id=None,  batch_count=1):
    """
    Incrementally update user metrics after each event.
    Falls back to full recalculation if increment fails.
    """
    try:
        from django.contrib.auth import get_user_model
        from organizations.models import Organization
        User = get_user_model()
        
        user = User.objects.get(id=user_id)
        org = Organization.objects.get(id=organization_id)
        
        # Get or create today's metrics
        today = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        metrics, created = UserActivityMetrics.objects.get_or_create(
            user=user,
            organization=org,
            project_id=project_id,
            period_start=today,
            period_end=today + timedelta(days=1),
            granularity='day',
            defaults={'last_activity': timezone.now()}
        )
        
        # Increment appropriate counter based on event type
        update_fields = ['last_activity', 'updated_at']
        
        if event_type == ActivityEventType.ANNOTATION_CREATE:
            metrics.annotations_created = F('annotations_created') + 1
            update_fields.extend(['annotations_created'])
            
            avg_time = ActivityEvent.objects.filter(
                user=user,
                organization=org,
                event_type=ActivityEventType.ANNOTATION_CREATE,
                timestamp__gte=today,
                timestamp__lt=today + timedelta(days=1),
                duration_ms__isnull=False,
                metadata__annotation_source='manual'
            ).aggregate(avg=Avg('duration_ms'))
            
            if avg_time['avg']:
                metrics.avg_annotation_time_seconds = round(avg_time['avg'] / 1000, 2)
                update_fields.append('avg_annotation_time_seconds')
            
            # Check if manual from latest event
            latest_event = ActivityEvent.objects.filter(
                user=user,
                event_type=ActivityEventType.ANNOTATION_CREATE,
                timestamp__gte=today
            ).order_by('-timestamp').first()
            
            if latest_event and latest_event.metadata.get('annotation_source') == 'manual':
                metrics.manual_annotations = F('manual_annotations') + batch_count
                update_fields.append('manual_annotations')
                
        elif event_type == ActivityEventType.ANNOTATION_UPDATE:
            metrics.annotations_updated = F('annotations_updated') + batch_count
            update_fields.append('annotations_updated')
            
        elif event_type == ActivityEventType.ANNOTATION_DELETE:
            metrics.annotations_deleted = F('annotations_deleted') + batch_count
            update_fields.append('annotations_deleted')
            
        elif event_type == ActivityEventType.IMAGE_REVIEW:
            metrics.images_reviewed = F('images_reviewed') + batch_count
            update_fields.append('images_reviewed')
            
        elif event_type == ActivityEventType.IMAGE_FINALIZE:
            metrics.images_finalized = F('images_finalized') + batch_count
            update_fields.append('images_finalized')
            
        elif event_type == ActivityEventType.PREDICTION_ACCEPT:
            metrics.ai_predictions_accepted = F('ai_predictions_accepted') + batch_count
            update_fields.append('ai_predictions_accepted')
            
        elif event_type == ActivityEventType.PREDICTION_EDIT:
            metrics.ai_predictions_edited = F('ai_predictions_edited') + batch_count
            update_fields.append('ai_predictions_edited')
            
        elif event_type == ActivityEventType.PREDICTION_REJECT:
            metrics.ai_predictions_rejected = F('ai_predictions_rejected') + batch_count
            update_fields.append('ai_predictions_rejected')
            
        elif event_type == ActivityEventType.IMAGE_UPLOAD:
            metrics.images_uploaded = F('images_uploaded') + batch_count
            update_fields.append('images_uploaded')
            
        elif event_type == ActivityEventType.IMAGE_ADD_TO_PROJECT:
            metrics.images_added_to_project = F('images_added_to_project') + batch_count
            update_fields.append('images_added_to_project')
            
        elif event_type == ActivityEventType.JOB_COMPLETE:
            metrics.jobs_completed = F('jobs_completed') + batch_count
            update_fields.append('jobs_completed')
        
        metrics.last_activity = timezone.now()
        metrics.save(update_fields=update_fields)
        
        logger.info(f"Updated metrics for user {user_id}, event {event_type}")
        
    except Exception as exc:
        logger.error(f"Failed to update user metrics: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


# ============================================================================
# SCHEDULED AGGREGATIONS
# ============================================================================

@shared_task
def aggregate_hourly_metrics():
    """
    Aggregate metrics every hour for the previous hour.
    Runs at :05 past each hour.
    """
    now = timezone.now()
    hour_start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    hour_end = hour_start + timedelta(hours=1)
    
    logger.info(f"Aggregating hourly metrics for {hour_start} to {hour_end}")
    
    # Get all active user-org-project combinations from last hour
    active_combinations = ActivityEvent.objects.filter(
        timestamp__gte=hour_start,
        timestamp__lt=hour_end
    ).values('user_id', 'organization_id', 'project_id').distinct()
    
    count = 0
    for combo in active_combinations:
        compute_user_metrics_for_period(
            user_id=combo['user_id'],
            organization_id=combo['organization_id'],
            project_id=combo['project_id'],
            start=hour_start,
            end=hour_end,
            granularity='hour'
        )
        count += 1
    
    logger.info(f"Aggregated {count} hourly metric records")
    return f"Processed {count} user-hour combinations"


@shared_task
def aggregate_daily_metrics():
    """
    Aggregate daily metrics at midnight for the previous day.
    Runs at 00:05 daily.
    """
    yesterday = timezone.now().date() - timedelta(days=1)
    start = timezone.datetime.combine(yesterday, timezone.datetime.min.time())
    start = timezone.make_aware(start)
    end = start + timedelta(days=1)
    
    logger.info(f"Aggregating daily metrics for {yesterday}")
    
    # Get all active users from yesterday
    active_combinations = ActivityEvent.objects.filter(
        timestamp__gte=start,
        timestamp__lt=end
    ).values('user_id', 'organization_id', 'project_id').distinct()
    
    count = 0
    for combo in active_combinations:
        compute_user_metrics_for_period(
            user_id=combo['user_id'],
            organization_id=combo['organization_id'],
            project_id=combo['project_id'],
            start=start,
            end=end,
            granularity='day'
        )
        count += 1
    
    # Also compute project-level metrics
    compute_all_project_metrics(start, end)
    
    logger.info(f"Aggregated {count} daily metric records")
    return f"Processed {count} user-day combinations"


@shared_task
def aggregate_weekly_metrics():
    """
    Aggregate weekly metrics every Monday at 01:00.
    """
    today = timezone.now().date()
    week_start = today - timedelta(days=today.weekday() + 7)  # Last Monday
    week_end = week_start + timedelta(days=7)
    
    start = timezone.datetime.combine(week_start, timezone.datetime.min.time())
    start = timezone.make_aware(start)
    end = timezone.datetime.combine(week_end, timezone.datetime.min.time())
    end = timezone.make_aware(end)
    
    logger.info(f"Aggregating weekly metrics for {week_start} to {week_end}")
    
    active_combinations = ActivityEvent.objects.filter(
        timestamp__gte=start,
        timestamp__lt=end
    ).values('user_id', 'organization_id', 'project_id').distinct()
    
    count = 0
    for combo in active_combinations:
        compute_user_metrics_for_period(
            user_id=combo['user_id'],
            organization_id=combo['organization_id'],
            project_id=combo['project_id'],
            start=start,
            end=end,
            granularity='week'
        )
        count += 1
    
    logger.info(f"Aggregated {count} weekly metric records")
    return f"Processed {count} user-week combinations"


@shared_task
def aggregate_monthly_metrics():
    """
    Aggregate monthly metrics on the 1st of each month at 02:00.
    """
    today = timezone.now().date()
    last_month = (today.replace(day=1) - timedelta(days=1))
    month_start = last_month.replace(day=1)
    month_end = today.replace(day=1)
    
    start = timezone.datetime.combine(month_start, timezone.datetime.min.time())
    start = timezone.make_aware(start)
    end = timezone.datetime.combine(month_end, timezone.datetime.min.time())
    end = timezone.make_aware(end)
    
    logger.info(f"Aggregating monthly metrics for {last_month.strftime('%B %Y')}")
    
    active_combinations = ActivityEvent.objects.filter(
        timestamp__gte=start,
        timestamp__lt=end
    ).values('user_id', 'organization_id', 'project_id').distinct()
    
    count = 0
    for combo in active_combinations:
        compute_user_metrics_for_period(
            user_id=combo['user_id'],
            organization_id=combo['organization_id'],
            project_id=combo['project_id'],
            start=start,
            end=end,
            granularity='month'
        )
        count += 1
    
    logger.info(f"Aggregated {count} monthly metric records")
    return f"Processed {count} user-month combinations"


# ============================================================================
# CORE COMPUTATION FUNCTIONS
# ============================================================================

def compute_user_metrics_for_period(user_id, organization_id, project_id, start, end, granularity):
    """
    Compute comprehensive metrics for a user in a given time period.
    """
    from django.contrib.auth import get_user_model
    User = get_user_model()
    
    if not user_id:
        return
    
    # Filter events
    events = ActivityEvent.objects.filter(
        user_id=user_id,
        organization_id=organization_id,
        timestamp__gte=start,
        timestamp__lt=end
    )
    
    if project_id:
        events = events.filter(project_id=project_id)
    
    # Aggregate annotations
    annotation_stats = events.filter(
        event_type__in=[
            ActivityEventType.ANNOTATION_CREATE,
            ActivityEventType.ANNOTATION_UPDATE,
            ActivityEventType.ANNOTATION_DELETE,
        ]
    ).aggregate(
        created=Count('event_id', filter=Q(event_type=ActivityEventType.ANNOTATION_CREATE)),
        updated=Count('event_id', filter=Q(event_type=ActivityEventType.ANNOTATION_UPDATE)),
        deleted=Count('event_id', filter=Q(event_type=ActivityEventType.ANNOTATION_DELETE)),
    )
    
    # Count manual annotations from metadata
    manual_count = events.filter(
        event_type=ActivityEventType.ANNOTATION_CREATE,
        metadata__annotation_source='manual'
    ).count()
    
    # Aggregate AI interaction
    ai_stats = events.filter(
        event_type__in=[
            ActivityEventType.PREDICTION_ACCEPT,
            ActivityEventType.PREDICTION_EDIT,
            ActivityEventType.PREDICTION_REJECT,
        ]
    ).aggregate(
        accepted=Count('event_id', filter=Q(event_type=ActivityEventType.PREDICTION_ACCEPT)),
        edited=Count('event_id', filter=Q(event_type=ActivityEventType.PREDICTION_EDIT)),
        rejected=Count('event_id', filter=Q(event_type=ActivityEventType.PREDICTION_REJECT)),
    )
    
    # Aggregate reviews
    review_stats = events.filter(
        event_type__in=[
            ActivityEventType.IMAGE_REVIEW,
            ActivityEventType.IMAGE_FINALIZE,
        ]
    ).aggregate(
        reviewed=Count('event_id', filter=Q(event_type=ActivityEventType.IMAGE_REVIEW)),
        finalized=Count('event_id', filter=Q(event_type=ActivityEventType.IMAGE_FINALIZE)),
    )
    
    # Check for approved/rejected from metadata
    approved = events.filter(
        event_type=ActivityEventType.IMAGE_REVIEW,
        metadata__approved=True
    ).count()
    
    rejected = events.filter(
        event_type=ActivityEventType.IMAGE_REVIEW,
        metadata__approved=False
    ).count()
    
    # Image uploads
    upload_stats = events.filter(
        event_type=ActivityEventType.IMAGE_UPLOAD
    ).aggregate(
        count=Count('event_id'),
        total_size=Sum(
            Cast(F('metadata__file_size_mb'), FloatField())
        )
    )
    
    # Get avg annotation time from duration_ms
    avg_duration = events.filter(
        event_type=ActivityEventType.ANNOTATION_CREATE,
        duration_ms__isnull=False
    ).aggregate(avg=Avg('duration_ms'))
    
    # Get avg edit magnitude
    avg_edit = events.filter(
        event_type=ActivityEventType.PREDICTION_EDIT,
        metadata__edit_magnitude__isnull=False
    ).aggregate(
        avg=Avg(
            Cast(F('metadata__edit_magnitude'), FloatField())
        )
    )
    
    # Get last activity
    last_activity = events.aggregate(last=Max('timestamp'))['last']
    
    # Update or create metrics
    UserActivityMetrics.objects.update_or_create(
        user_id=user_id,
        organization_id=organization_id,
        project_id=project_id,
        period_start=start,
        granularity=granularity,
        defaults={
            'period_end': end,
            'annotations_created': annotation_stats['created'] or 0,
            'annotations_updated': annotation_stats['updated'] or 0,
            'annotations_deleted': annotation_stats['deleted'] or 0,
            'manual_annotations': manual_count,
            'ai_predictions_accepted': ai_stats['accepted'] or 0,
            'ai_predictions_edited': ai_stats['edited'] or 0,
            'ai_predictions_rejected': ai_stats['rejected'] or 0,
            'images_reviewed': review_stats['reviewed'] or 0,
            'images_approved': approved,
            'images_rejected': rejected,
            'images_finalized': review_stats['finalized'] or 0,
            'images_uploaded': upload_stats['count'] or 0,
            'total_upload_size_mb': upload_stats['total_size'] or 0,
            'avg_annotation_time_seconds': (
                avg_duration['avg'] / 1000 if avg_duration['avg'] else None
            ),
            'avg_edit_magnitude': avg_edit['avg'],
            'last_activity': last_activity or timezone.now(),
        }
    )


@shared_task
def compute_all_project_metrics(start_date=None, end_date=None):
    """
    Compute project-level metrics for all active projects.
    """
    from projects.models import Project
    
    if not start_date:
        yesterday = timezone.now().date() - timedelta(days=1)
        start_date = timezone.datetime.combine(yesterday, timezone.datetime.min.time())
        start_date = timezone.make_aware(start_date)
    
    if not end_date:
        end_date = start_date + timedelta(days=1)
    
    active_projects = Project.objects.filter(
        is_active=True,
        is_deleted=False
    )
    
    count = 0
    for project in active_projects:
        compute_project_metrics.delay(project.id, start_date)
        count += 1
    
    logger.info(f"Queued project metrics computation for {count} projects")
    return f"Queued {count} projects"


@shared_task(bind=True, queue="activity", name="activity:compute_project_metrics")
def compute_project_metrics(self, project_id, start_date=None):
    """
    Compute project-level metrics including untouched predictions.
    """
    from projects.models import Project, ProjectImage
    from annotations.models import Annotation
    
    try:
        project = Project.objects.get(id=project_id)
    except Project.DoesNotExist:
        logger.error(f"Project {project_id} not found")
        return
    
    if not start_date:
        start_date = timezone.now().date() - timedelta(days=1)
        start_date = timezone.datetime.combine(start_date, timezone.datetime.min.time())
        start_date = timezone.make_aware(start_date)
    
    end_date = start_date + timedelta(days=1)
    
    # Count untouched predictions (version=1, source=prediction, not edited)
    untouched = Annotation.objects.filter(
        project_image__project=project,
        annotation_source='prediction',
        version=1,
        is_active=True,
        created_at__lt=end_date
    ).count()
    
    # Count edited predictions
    edited = Annotation.objects.filter(
        project_image__project=project,
        annotation_source='prediction',
        version__gt=1,
        is_active=True,
        updated_at__gte=start_date,
        updated_at__lt=end_date
    ).count()
    
    # Count rejected (soft deleted predictions)
    rejected = Annotation.objects.filter(
        project_image__project=project,
        annotation_source='prediction',
        is_deleted=True,
        deleted_at__gte=start_date,
        deleted_at__lt=end_date
    ).count()
    
    # Image status breakdown
    status_counts = ProjectImage.objects.filter(
        project=project,
        is_active=True
    ).aggregate(
        total=Count('id'),
        unannotated=Count('id', filter=Q(status='unannotated')),
        annotated=Count('id', filter=Q(status='annotated')),
        reviewed=Count('id', filter=Q(status='reviewed')),
        finalized=Count('id', filter=Q(finalized=True)),
    )
    
    # Total annotations
    total_annotations = Annotation.objects.filter(
        project_image__project=project,
        is_active=True
    ).count()
    
    manual_annotations = Annotation.objects.filter(
        project_image__project=project,
        annotation_source='manual',
        is_active=True
    ).count()
    
    ai_predictions = Annotation.objects.filter(
        project_image__project=project,
        annotation_source='prediction',
        is_active=True
    ).count()
    
    # Avg annotations per image
    avg_annotations = total_annotations / status_counts['total'] if status_counts['total'] > 0 else 0
    
    # Active contributors in period
    active_users = ActivityEvent.objects.filter(
        project=project,
        timestamp__gte=start_date,
        timestamp__lt=end_date
    ).values('user_id').distinct().count()
    
    # Top contributor
    top_contributor = ActivityEvent.objects.filter(
        project=project,
        timestamp__gte=start_date,
        timestamp__lt=end_date
    ).values('user_id').annotate(
        event_count=Count('event_id')
    ).order_by('-event_count').first()
    
    # Annotations per hour (from events in this period)
    period_hours = (end_date - start_date).total_seconds() / 3600
    annotations_in_period = ActivityEvent.objects.filter(
        project=project,
        event_type=ActivityEventType.ANNOTATION_CREATE,
        timestamp__gte=start_date,
        timestamp__lt=end_date
    ).count()
    
    annotations_per_hour = annotations_in_period / period_hours if period_hours > 0 else 0
    
    # Update metrics
    ProjectActivityMetrics.objects.update_or_create(
        project=project,
        organization=project.organization,
        period_start=start_date,
        granularity='day',
        defaults={
            'period_end': end_date,
            'total_images': status_counts['total'],
            'images_unannotated': status_counts['unannotated'],
            'images_annotated': status_counts['annotated'],
            'images_reviewed': status_counts['reviewed'],
            'images_finalized': status_counts['finalized'],
            'total_annotations': total_annotations,
            'manual_annotations': manual_annotations,
            'ai_predictions': ai_predictions,
            'untouched_predictions': untouched,
            'edited_predictions': edited,
            'rejected_predictions': rejected,
            'avg_annotations_per_image': round(avg_annotations, 2),
            'annotations_per_hour': round(annotations_per_hour, 2),
            'active_users': active_users,
            'top_contributor_id': top_contributor['user_id'] if top_contributor else None,
        }
    )
    
    logger.info(f"Computed metrics for project {project.name}")


# ============================================================================
# CLEANUP TASKS
# ============================================================================

@shared_task
def cleanup_old_sessions():
    """
    Clean up inactive sessions older than 24 hours.
    Runs every hour.
    """
    cutoff = timezone.now() - timedelta(hours=24)
    deleted_count, _ = UserSessionActivity.objects.filter(
        last_activity__lt=cutoff
    ).delete()
    
    logger.info(f"Deleted {deleted_count} old sessions")
    return f"Deleted {deleted_count} old sessions"


@shared_task
def cleanup_old_events():
    """
    Archive or delete events older than retention period (90 days).
    Runs daily at 03:00.
    """
    retention_days = 90
    cutoff = timezone.now() - timedelta(days=retention_days)
    
    # Option 1: Delete (if you don't need long-term history)
    # deleted_count, _ = ActivityEvent.objects.filter(timestamp__lt=cutoff).delete()
    
    # Option 2: Archive to separate table or cold storage
    old_events = ActivityEvent.objects.filter(timestamp__lt=cutoff)
    count = old_events.count()
    
    # TODO: Implement archiving logic here
    # For now, just log
    logger.info(f"Found {count} events older than {retention_days} days for archiving")
    
    return f"Found {count} events for archiving"


@shared_task
def refresh_materialized_views():
    """
    Refresh PostgreSQL materialized views.
    Runs every 30 minutes.
    """
    from django.db import connection
    
    with connection.cursor() as cursor:
        try:
            cursor.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY org_activity_summary;")
            logger.info("Refreshed org_activity_summary materialized view")
            return "Materialized views refreshed"
        except Exception as e:
            logger.error(f"Failed to refresh materialized views: {e}")
            raise


# ============================================================================
# REPORTING TASKS
# ============================================================================

@shared_task
def generate_weekly_reports():
    """
    Generate weekly activity reports for all organizations.
    Runs every Monday at 06:00.
    """
    from organizations.models import Organization
    
    last_week_start = timezone.now().date() - timedelta(days=7)
    last_week_end = timezone.now().date()
    
    start = timezone.datetime.combine(last_week_start, timezone.datetime.min.time())
    start = timezone.make_aware(start)
    end = timezone.datetime.combine(last_week_end, timezone.datetime.min.time())
    end = timezone.make_aware(end)
    
    count = 0
    for org in Organization.objects.filter(is_active=True):
        generate_organization_report.delay(
            organization_id=org.id,
            start_date=start,
            end_date=end,
            report_type='weekly'
        )
        count += 1
    
    logger.info(f"Queued weekly reports for {count} organizations")
    return f"Queued {count} weekly reports"


@shared_task
def generate_organization_report(organization_id, start_date, end_date, report_type='weekly'):
    """
    Generate comprehensive activity report for an organization.
    Can be extended to send emails, generate PDFs, etc.
    """
    from organizations.models import Organization
    
    try:
        org = Organization.objects.get(id=organization_id)
    except Organization.DoesNotExist:
        logger.error(f"Organization {organization_id} not found")
        return
    
    # Gather metrics
    user_metrics = UserActivityMetrics.objects.filter(
        organization=org,
        period_start__gte=start_date,
        period_end__lte=end_date
    ).aggregate(
        total_annotations=Sum('annotations_created'),
        total_reviews=Sum('images_reviewed'),
        total_uploads=Sum('images_uploaded'),
        active_users=Count('user_id', distinct=True)
    )
    
    project_metrics = ProjectActivityMetrics.objects.filter(
        organization=org,
        period_start__gte=start_date,
        period_end__lte=end_date
    ).aggregate(
        total_images=Sum('total_images'),
        total_finalized=Sum('images_finalized'),
    )
    
    # Store report or send email
    report_data = {
        'organization': org.name,
        'period': f"{start_date.date()} to {end_date.date()}",
        'type': report_type,
        'metrics': {
            **user_metrics,
            **project_metrics
        }
    }
    
    # TODO: Implement email sending or PDF generation
    logger.info(f"Generated {report_type} report for {org.name}: {report_data}")
    
    return report_data


# ============================================================================
# HEALTH CHECK
# ============================================================================

@shared_task
def health_check_activity_system():
    """
    Check activity tracking system health.
    Runs every 15 minutes.
    """
    issues = []
    
    # Check event lag
    latest_event = ActivityEvent.objects.order_by('-timestamp').first()
    if latest_event:
        lag = (timezone.now() - latest_event.timestamp).total_seconds()
        if lag > 300:  # 5 minutes
            issues.append(f"Event lag: {lag}s")
    
    # Check aggregation freshness
    latest_metric = UserActivityMetrics.objects.order_by('-updated_at').first()
    if latest_metric:
        staleness = (timezone.now() - latest_metric.updated_at).total_seconds()
        if staleness > 7200:  # 2 hours
            issues.append(f"Metrics staleness: {staleness}s")
    
    # Check for failed tasks (requires celery result backend)
    # TODO: Implement celery task failure monitoring
    
    if issues:
        logger.warning(f"Activity system health issues: {issues}")
        # TODO: Send alerts to monitoring system
    else:
        logger.info("Activity system health check passed")
    
    return {
        'status': 'unhealthy' if issues else 'healthy',
        'issues': issues,
        'timestamp': timezone.now().isoformat()
    }
    
@shared_task(bind=True, max_retries=3)
def create_batch_finalize_events(self, project_id, project_image_ids, user_id):
    """Create activity events for batch finalized images."""
    try:
        from projects.models import Project, ProjectImage
        from django.contrib.auth import get_user_model
        from activity.signals import create_activity_event
        from activity.models import ActivityEventType
        import uuid
        
        User = get_user_model()
        
        project = Project.objects.get(project_id=project_id)
        user = User.objects.get(id=user_id)
        session_id = uuid.uuid4()
        
        # Batch create events
        events = []
        for img_id in project_image_ids:
            events.append(
                create_activity_event(
                    organization=project.organization,
                    event_type=ActivityEventType.IMAGE_FINALIZE,
                    user=user,
                    project=project,
                    session_id=session_id,
                    metadata={
                        'project_image_id': img_id,
                        'batch_operation': True,
                        'batch_size': len(project_image_ids)
                    }
                )
            )
        
        logger.info(f"Created {len(events)} finalize events for batch operation")
        
        # Update user metrics (single update for entire batch)
        update_user_metrics_async.delay(
            user_id=user_id,
            organization_id=str(project.organization_id),
            event_type=ActivityEventType.IMAGE_FINALIZE,
            project_id=project.id,
            batch_count=len(project_image_ids)  # Pass batch size
        )
        
    except Exception as exc:
        logger.error(f"Failed to create batch events: {exc}")
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)