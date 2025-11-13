from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import List, Optional
from datetime import date
from datetime import datetime, timedelta
from pydantic import BaseModel
from django.utils import timezone
from django.db.models import Sum, Avg, Count, Q, Max
from organizations.models import Organization
from activity.models import (
    OrgActivitySummary,
    ActivityEventType,
    UserActivityMetrics,
    UserSessionActivity,
    ProjectActivityMetrics,
    ActivityEvent,
)

from api.routers.activity.schemas import (
    UserActivitySummary,
    ProjectProgressSummary,
    LeaderboardEntry,
    ActivityTimelineEvent,
    ActivityTrendPoint,
    PredictionQualityMetrics,
    OrganizationActivitySummary,
    OrgActivitySummaryResponse
)
from uuid import UUID
from asgiref.sync import sync_to_async
from api.dependencies import RequestContext, get_request_context

router = APIRouter(prefix="/activity",)

@router.get("/organization/summary", response_model=OrganizationActivitySummary)
async def get_organization_activity_summary(
    user_id: Optional[str] = Query(None, description="Filter by specific user"),
    project_id: Optional[UUID] = Query(None, description="Filter by specific project"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get organization-wide activity summary with optional filters.
    
    **Use Case**: Executive dashboard, organization overview
    **Filters**: Can narrow down by user or project
    """
    @sync_to_async
    def fetch_org_summary(org, user_id, project_id, start_date, end_date):
        
        if not start_date:
            start_date = timezone.now() - timedelta(days=30)
        if not end_date:
            end_date = timezone.now()
        
        # Build base query
        metrics_query = UserActivityMetrics.objects.filter(
            organization=org,
            period_start__gte=start_date,
            period_end__lte=end_date
        )
        
        # Apply filters
        if user_id:
            metrics_query = metrics_query.filter(user_id=user_id)
        if project_id:
            metrics_query = metrics_query.filter(project__project_id=project_id)
        
        # Aggregate metrics
        summary = metrics_query.aggregate(
            total_annotations=Sum('annotations_created'),
            manual_annotations=Sum('manual_annotations'),
            ai_edited=Sum('ai_predictions_edited'),
            ai_accepted=Sum('ai_predictions_accepted'),
            images_reviewed=Sum('images_reviewed'),
            images_finalized=Sum('images_finalized'),
            avg_time=Avg('avg_annotation_time_seconds'),
            avg_edit_mag=Avg('avg_edit_magnitude'),
            total_reviews=Sum('images_reviewed'),
            total_uploads=Sum('images_uploaded'),
            active_users=Count('user_id', distinct=True),
    
        )
        
        # Event counts
        events_query = ActivityEvent.objects.filter(
            organization=org,
            timestamp__gte=start_date,
            timestamp__lt=end_date
        )
        
        if user_id:
            events_query = events_query.filter(user_id=user_id)
        if project_id:
            events_query = events_query.filter(project__project_id=project_id)
        
        total_events = events_query.count()
        
        # Project stats
        project_metrics_query = ProjectActivityMetrics.objects.filter(
            organization=org,
            period_start__gte=start_date,
            period_end__lte=end_date
        )
        
        if project_id:
            project_metrics_query = project_metrics_query.filter(project__project_id=project_id)
        
        project_stats = project_metrics_query.aggregate(
            total_projects=Count('project_id', distinct=True),
            active_projects=Count('project_id', filter=Q(active_users__gt=0), distinct=True)
        )
        
        # Top performers (if not filtered by user)
        top_annotator = None
        top_reviewer = None
        
        if not user_id:
            top_annotator_data = metrics_query.values(
                'user__id', 'user__username', 'user__first_name', 'user__last_name'
            ).annotate(
                total=Sum('annotations_created')
            ).order_by('-total').first()
            
            if top_annotator_data:
                top_annotator = {
                    'user_id': str(top_annotator_data['user__id']),
                    'username': top_annotator_data['user__username'],
                    'full_name': f"{top_annotator_data['user__first_name']} {top_annotator_data['user__last_name']}".strip(),
                    'count': top_annotator_data['total']
                }
            
            top_reviewer_data = metrics_query.values(
                'user__id', 'user__username', 'user__first_name', 'user__last_name'
            ).annotate(
                total=Sum('images_reviewed')
            ).order_by('-total').first()
            
            if top_reviewer_data:
                top_reviewer = {
                    'user_id': str(top_reviewer_data['user__id']),
                    'username': top_reviewer_data['user__username'],
                    'full_name': f"{top_reviewer_data['user__first_name']} {top_reviewer_data['user__last_name']}".strip(),
                    'count': top_reviewer_data['total']
                }
        
        return OrganizationActivitySummary(
            organization_id=str(org.id),
            organization_name=org.name,
            period={
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            total_events=total_events,
            total_annotations=summary['total_annotations'] or 0,
            manual_annotations=summary['manual_annotations'] or 0,
            ai_predictions_edited=summary['ai_edited'] or 0,
            ai_predictions_accepted=summary['ai_accepted'] or 0,
            images_reviewed=summary['images_reviewed'] or 0,
            images_finalized=summary['images_finalized'] or 0,
            avg_annotation_time_seconds=summary['avg_time'],
            avg_edit_magnitude=summary['avg_edit_mag'],
        
            total_active_users=summary['active_users'] or 0,
            avg_annotations_per_user=round(
                (summary['total_annotations'] or 0) / (summary['active_users'] or 1), 2
            ),
            
            total_projects=project_stats['total_projects'] or 0,
            active_projects=project_stats['active_projects'] or 0,
            top_annotator=top_annotator,
            top_reviewer=top_reviewer
        )
    
    return await fetch_org_summary(ctx.organization, user_id, project_id, start_date, end_date)


@router.get("/organization/users", response_model=List[UserActivitySummary])
async def get_organization_users_activity(
    project_id: Optional[UUID] = Query(None, description="Filter by project"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    sort_by: str = Query('total_annotations', regex='^(total_annotations|images_reviewed|images_finalized)$'),
    limit: int = Query(50, ge=1, le=200),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get all users' activity in organization with optional project filter.
    
    **Use Case**: Team performance dashboard, resource allocation
    """
    @sync_to_async
    def fetch_users_activity(org: Organization, project_id, start_date, end_date, sort_by, limit):        
        
        if not start_date:
            start_date = timezone.now() - timedelta(days=30)
        if not end_date:
            end_date = timezone.now()
        
        # Build query
        metrics_query = UserActivityMetrics.objects.filter(
            organization=org,
            period_start__gte=start_date,
            period_end__lte=end_date
        )
        
        if project_id:
            metrics_query = metrics_query.filter(project__project_id=project_id)
        
        # Aggregate by user
        users_data = metrics_query.values(
            'user__id', 'user__username', 'user__first_name', 'user__last_name'
        ).annotate(
            total_annotations=Sum('annotations_created'),
            manual_annotations=Sum('manual_annotations'),
            ai_edited=Sum('ai_predictions_edited'),
            ai_accepted=Sum('ai_predictions_accepted'),
            images_reviewed=Sum('images_reviewed'),
            images_finalized=Sum('images_finalized'),
            avg_time=Avg('avg_annotation_time_seconds'),
            avg_edit_mag=Avg('avg_edit_magnitude'),
            last_activity=Max('last_activity')
        )
        
        # Sort
        sort_field = {
            'total_annotations': '-total_annotations',
            'images_reviewed': '-images_reviewed',
            'images_finalized': '-images_finalized'
        }.get(sort_by, '-total_annotations')
        
        users_data = users_data.order_by(sort_field)[:limit]
        
        # Get session counts for each user
        result = []
        for user_data in users_data:
            session_count = ActivityEvent.objects.filter(
                user_id=user_data['user__id'],
                organization=org,
                timestamp__gte=start_date,
                timestamp__lte=end_date
            ).values('session_id').distinct().count()
            
            full_name = f"{user_data['user__first_name']} {user_data['user__last_name']}".strip()
            
            result.append(UserActivitySummary(
                user_id=str(user_data['user__id']),
                username=user_data['user__username'],
                full_name=full_name or user_data['user__username'],
                total_annotations=user_data['total_annotations'] or 0,
                manual_annotations=user_data['manual_annotations'] or 0,
                ai_predictions_edited=user_data['ai_edited'] or 0,
                ai_predictions_accepted=user_data['ai_accepted'] or 0,
                images_reviewed=user_data['images_reviewed'] or 0,
                images_finalized=user_data['images_finalized'] or 0,
                avg_annotation_time_seconds=user_data['avg_time'],
                avg_edit_magnitude=user_data['avg_edit_mag'],
                last_activity=user_data['last_activity'],
                total_sessions=session_count
            ))
        
        return result
    
    return await fetch_users_activity(ctx.organization, project_id, start_date, end_date, sort_by, limit)


@router.get("/organization/projects", response_model=List[ProjectProgressSummary])
async def get_organization_projects_progress(
    user_id: Optional[str] = Query(None, description="Filter by user contribution"),
    status: Optional[str] = Query(None, regex='^(active|completed|stalled)$'),
    limit: int = Query(50, ge=1, le=200),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get all projects progress in organization with optional filters.
    
    **Use Case**: Portfolio view, project prioritization
    **Filters**: 
    - user_id: Show only projects where user has contributed
    - status: active (recent activity), completed (>90% done), stalled (no activity 7+ days)
    """
    @sync_to_async
    def fetch_projects_progress(org: Organization, user_id, status, limit):        
        # Get latest metrics for each project
        latest_metrics = ProjectActivityMetrics.objects.filter(
            organization=org,
            granularity='day'
        ).values('project_id').annotate(
            latest_date=Max('period_start')
        )
        
        project_ids = [m['project_id'] for m in latest_metrics]
        
        projects_data = []
        for project_id in project_ids:
            metric = ProjectActivityMetrics.objects.filter(
                project_id=project_id,
                granularity='day'
            ).order_by('-period_start').first()
            
            if not metric:
                continue
            
            # Filter by user contribution
            if user_id:
                user_contributed = ActivityEvent.objects.filter(
                    project_id=project_id,
                    user_id=user_id
                ).exists()
                if not user_contributed:
                    continue
            
            # Calculate status
            completion = (metric.images_finalized / metric.total_images * 100) if metric.total_images > 0 else 0
            days_since_activity = (timezone.now().date() - metric.period_start.date()).days
            
            project_status = 'active'
            if completion > 90:
                project_status = 'completed'
            elif days_since_activity > 7:
                project_status = 'stalled'
            
            # Filter by status
            if status and project_status != status:
                continue
            
            total_predictions = (
                metric.untouched_predictions +
                metric.edited_predictions +
                metric.rejected_predictions
            )
            acceptance = (
                (metric.untouched_predictions + metric.edited_predictions) /
                total_predictions * 100 if total_predictions > 0 else 0
            )
            
            projects_data.append(ProjectProgressSummary(
                project_id=str(metric.project.project_id),
                project_name=metric.project.name,
                total_images=metric.total_images,
                images_unannotated=metric.images_unannotated,
                images_annotated=metric.images_annotated,
                images_reviewed=metric.images_reviewed,
                images_finalized=metric.images_finalized,
                completion_percentage=round(completion, 2),
                untouched_predictions=metric.untouched_predictions,
                edited_predictions=metric.edited_predictions,
                rejected_predictions=metric.rejected_predictions,
                prediction_acceptance_rate=round(acceptance, 2),
                annotations_per_hour=float(metric.annotations_per_hour) if metric.annotations_per_hour else None,
                active_contributors=metric.active_users
            ))
        
        # Sort by completion percentage
        projects_data.sort(key=lambda x: x.completion_percentage, reverse=True)
        
        return projects_data[:limit]
    
    return await fetch_projects_progress(ctx.organization, user_id, status, limit)

# apps/activity/api/endpoints.py

@router.get("/organization/timeline", response_model=List[ActivityTimelineEvent])
async def get_organization_timeline(
    user_id: Optional[str] = Query(None, description="Filter by user"),
    project_id: Optional[UUID] = Query(None, description="Filter by project"),
    event_types: Optional[List[str]] = Query(None, description="Filter by event types"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get organization-wide activity timeline with filters.
    
    **Use Case**: Audit logs, activity monitoring
    **Filters**: user_id, project_id, event_types, date range
    """
    @sync_to_async
    def fetch_org_timeline(org: Organization, user_id, project_id, event_types, start_date, end_date, limit, offset):
        
        # Build query
        events = ActivityEvent.objects.filter(organization=org)
        
        if user_id:
            events = events.filter(user_id=user_id)
        
        if project_id:
            events = events.filter(project__project_id=project_id)
        
        if event_types:
            events = events.filter(event_type__in=event_types)
        
        if start_date:
            events = events.filter(timestamp__gte=start_date)
        
        if end_date:
            events = events.filter(timestamp__lte=end_date)
        
        # Paginate
        events = events.select_related('user', 'project').order_by('-timestamp')[offset:offset+limit]
        
        # Format response
        timeline = []
        for event in events:
            timeline.append(ActivityTimelineEvent(
                event_id=str(event.event_id),
                event_type=event.event_type,
                user=event.user.username if event.user else 'System',
                timestamp=event.timestamp,
                project=event.project.name if event.project else None,
                metadata=event.metadata
            ))
        
        return timeline
    
    return await fetch_org_timeline(
        ctx.organization, user_id, project_id, event_types, start_date, end_date, limit, offset
    )


@router.get("/organization/leaderboard", response_model=List[LeaderboardEntry])
async def get_organization_leaderboard(
    project_id: Optional[UUID] = Query(None, description="Filter by project"),
    metric: str = Query('annotations_created', regex='^(annotations_created|images_reviewed|images_finalized)$'),
    period_days: int = Query(30, ge=1, le=365),
    limit: int = Query(10, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get organization-wide leaderboard with optional project filter.
    
    **Use Case**: Team motivation, performance tracking
    **Supported Metrics**: annotations_created, images_reviewed, images_finalized
    """
    @sync_to_async
    def fetch_org_leaderboard(org: Organization, project_id, metric, period_days, limit):
        start_date = timezone.now() - timedelta(days=period_days)
        
        # Build query
        metrics_query = UserActivityMetrics.objects.filter(
            organization=org,
            period_start__gte=start_date
        )
        
        if project_id:
            metrics_query = metrics_query.filter(project__project_id=project_id)
        
        # Aggregate by user
        leaderboard_data = metrics_query.values(
            'user__id', 'user__username', 'user__first_name', 'user__last_name'
        ).annotate(
            total=Sum(metric)
        ).order_by('-total')[:limit]
        
        # Calculate total for percentages
        grand_total = sum(entry['total'] or 0 for entry in leaderboard_data)
        
        # Format response
        leaderboard = []
        for rank, entry in enumerate(leaderboard_data, 1):
            full_name = f"{entry['user__first_name']} {entry['user__last_name']}".strip()
            leaderboard.append(LeaderboardEntry(
                rank=rank,
                user_id=str(entry['user__id']),
                username=entry['user__username'],
                full_name=full_name or entry['user__username'],
                metric_value=entry['total'] or 0,
                percentage_of_total=round((entry['total'] / grand_total * 100) if grand_total > 0 else 0, 2)
            ))
        
        return leaderboard
    
    return await fetch_org_leaderboard(ctx.organization, project_id, metric, period_days, limit)

@router.get("/organization/heatmap")
async def get_activity_heatmap(
    start_date: datetime,
    end_date: datetime,
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get hourly activity heatmap for visualization.
    
    **Response Format**: 
```json
    {
        "2025-01-15": {
            "00": 5, "01": 2, ..., "23": 12
        },
        ...
    }
```
    """
    @sync_to_async
    def fetch_activity_heatmap(ctx: RequestContext, start_date: datetime, end_date: datetime):
        from django.db.models.functions import TruncHour, TruncDate
        
        # Aggregate by hour
        hourly_counts = ActivityEvent.objects.filter(
            organization=ctx.organization,
            timestamp__gte=start_date,
            timestamp__lte=end_date
        ).annotate(
            date=TruncDate('timestamp'),
            hour=TruncHour('timestamp')
        ).values('date', 'hour').annotate(
            count=Count('event_id')
        ).order_by('date', 'hour')
        
        # Format as nested dict
        heatmap = {}
        for entry in hourly_counts:
            date_str = entry['date'].isoformat()
            hour_str = entry['hour'].strftime('%H')
            
            if date_str not in heatmap:
                heatmap[date_str] = {}
            
            heatmap[date_str][hour_str] = entry['count']
        
        return heatmap

    return await fetch_activity_heatmap(
        ctx, start_date, end_date
    )

@router.get("/organization/activity-trend", response_model=List[ActivityTrendPoint])
async def get_organization_activity_trend(
    user_id: Optional[str] = Query(None, description="Filter by user"),
    project_id: Optional[UUID] = Query(None, description="Filter by project"),
    days: int = Query(30, ge=7, le=365, description="Number of days"),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get daily activity trend for charts.
    
    **Use Case**: Line charts, trend visualization
    **Returns**: Daily aggregated metrics for the specified period
    """
    @sync_to_async
    def fetch_activity_trend(org: Organization, user_id, project_id, days):
        from django.db.models.functions import TruncDate
        
        start_date = timezone.now() - timedelta(days=days)
        
        # Build query
        events_query = ActivityEvent.objects.filter(
            organization=org,
            timestamp__gte=start_date
        )
        
        if user_id:
            events_query = events_query.filter(user_id=user_id)
        
        if project_id:
            events_query = events_query.filter(project__project_id=project_id)
        
        # Aggregate by date
        daily_data = events_query.annotate(
            date=TruncDate('timestamp')
        ).values('date').annotate(
            annotations=Count('event_id', filter=Q(event_type=ActivityEventType.ANNOTATION_CREATE)),
            reviews=Count('event_id', filter=Q(event_type=ActivityEventType.IMAGE_REVIEW)),
            uploads=Count('event_id', filter=Q(event_type=ActivityEventType.IMAGE_UPLOAD)),
            active_users=Count('user_id', distinct=True)
        ).order_by('date')
        
        # Format response
        trend = []
        for entry in daily_data:
            trend.append(ActivityTrendPoint(
                date=entry['date'].strftime('%b %d'),
                annotations=entry['annotations'],
                reviews=entry['reviews'],
                uploads=entry['uploads'],
                active_users=entry['active_users']
            ))
        
        return trend
    
    return await fetch_activity_trend(ctx.organization, user_id, project_id, days)



@router.get(
    "/organizations/daily-summary",
    response_model=List[OrgActivitySummaryResponse],
    summary="Get Daily Activity Summary"
)
async def get_organization_daily_summary(
    days: int = Query(30, ge=1, le=90, description="Number of days to retrieve"),
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Get pre-aggregated daily activity summary from materialized view.
    
    **Performance**: Ultra-fast queries on pre-computed data.
    **Data**: Refreshed every 30 minutes.
    """
    from activity.models import OrgActivitySummary
    
    @sync_to_async
    def get_summary(ctx:RequestContext):
        cutoff_date = timezone.now().date() - timedelta(days=days)
        
        summaries = OrgActivitySummary.objects.filter(
            organization=ctx.organization,
            date__gte=cutoff_date
        ).order_by('-date')[:days]
        
        return list(summaries)
    
    summaries = await get_summary()
    
    return [
        OrgActivitySummaryResponse(
            date=s.date,
            total_events=s.total_events,
            active_users=s.active_users,
            annotation_events=s.annotation_events,
            image_events=s.image_events,
            annotations_created=s.annotations_created,
            images_reviewed=s.images_reviewed,
            images_uploaded=s.images_uploaded
        )
        for s in summaries
    ]


@router.get(
    "/organizations/activity-stats",
    summary="Get Organization Activity Statistics"
)
async def get_organization_activity_stats(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Get aggregated statistics for organization from materialized view.
    
    **Use Case**: Dashboard KPIs, executive reports.
    """
    from activity.models import OrgActivitySummary
    from django.db.models import Sum, Avg, Max
    
    if not start_date:
        start_date = timezone.now().date() - timedelta(days=30)
    if not end_date:
        end_date = timezone.now().date()
    
    @sync_to_async
    def get_stats(ctx:RequestContext):
        stats = OrgActivitySummary.objects.filter(
            organization=ctx.organization,
            date__gte=start_date,
            date__lte=end_date
        ).aggregate(
            total_events=Sum('total_events'),
            total_annotations=Sum('annotations_created'),
            total_reviews=Sum('images_reviewed'),
            total_uploads=Sum('images_uploaded'),
            avg_daily_users=Avg('active_users'),
            max_daily_users=Max('active_users'),
            days_with_activity=Count('date')
        )
        return stats
    
    stats = await get_stats()
    
    return {
        'organization_id': ctx.organization.id,
        'period': {
            'start': start_date.isoformat(),
            'end': end_date.isoformat()
        },
        'statistics': {
            'total_events': stats['total_events'] or 0,
            'total_annotations': stats['total_annotations'] or 0,
            'total_reviews': stats['total_reviews'] or 0,
            'total_uploads': stats['total_uploads'] or 0,
            'avg_daily_active_users': round(stats['avg_daily_users'] or 0, 1),
            'peak_daily_users': stats['max_daily_users'] or 0,
            'days_with_activity': stats['days_with_activity'] or 0
        }
    }


@router.post(
    "/admin/refresh-materialized-views",
    summary="Manually Refresh Materialized Views"
)
async def manually_refresh_materialized_views(
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Manually trigger materialized view refresh.
    
    **Requires**: Admin permissions.
    **Use Case**: Force refresh after bulk data import.
    """
    # Check admin permissions
    if not ctx.user.is_staff:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    from api.tasks.activity import refresh_materialized_views
    
    # Trigger async refresh
    task = refresh_materialized_views.delay()
    
    return {
        'status': 'queued',
        'task_id': task.id,
        'message': 'Materialized view refresh queued'
    }