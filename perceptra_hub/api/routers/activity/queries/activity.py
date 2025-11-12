# apps/activity/api/endpoints.py
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
from django.utils import timezone
from django.db.models import Sum, Avg, Count, Q, Max
from activity.models import (
    ActivityEvent,
    UserActivityMetrics,
    ProjectActivityMetrics,
    ActivityEventType
)

from uuid import UUID
from asgiref.sync import sync_to_async
from api.dependencies import RequestContext, get_request_context

router = APIRouter(prefix="/activity",)


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class UserActivitySummary(BaseModel):
    user_id: str
    username: str
    full_name: str
    
    # Annotation metrics
    total_annotations: int
    manual_annotations: int
    ai_predictions_edited: int
    ai_predictions_accepted: int
    
    # Review metrics
    images_reviewed: int
    images_finalized: int
    
    # Quality metrics
    avg_annotation_time_seconds: Optional[float]
    avg_edit_magnitude: Optional[float]
    
    # Activity
    last_activity: Optional[datetime]
    total_sessions: int


class ProjectProgressSummary(BaseModel):
    project_id: str
    project_name: str
    
    # Progress breakdown
    total_images: int
    images_unannotated: int
    images_annotated: int
    images_reviewed: int
    images_finalized: int
    completion_percentage: float
    
    # Quality insights
    untouched_predictions: int
    edited_predictions: int
    rejected_predictions: int
    prediction_acceptance_rate: float
    
    # Velocity
    annotations_per_hour: Optional[float]
    active_contributors: int


class ActivityTimelineEvent(BaseModel):
    event_id: str
    event_type: str
    user: str
    timestamp: datetime
    project: Optional[str]
    metadata: dict


class LeaderboardEntry(BaseModel):
    rank: int
    user_id: str
    username: str
    full_name: str
    metric_value: int
    percentage_of_total: float


class PredictionQualityMetrics(BaseModel):
    total_predictions: int
    untouched: int
    accepted_without_edit: int
    minor_edits: int
    major_edits: int
    class_changes: int
    rejected: int
    
    # Percentages
    untouched_percentage: float
    acceptance_rate: float
    avg_edit_magnitude: Optional[float]


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/users/{user_id}/summary", response_model=UserActivitySummary)
async def get_user_activity_summary(
    user_id: str,
    project_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    ctx: RequestContext = Depends(get_request_context)  # Your auth dependency
):
    """
    Get comprehensive activity summary for a user.
    
    **Optimized Query**: Uses pre-aggregated metrics table.
    """
    @sync_to_async
    def fetch_user_activity_summary(
        ctx: RequestContext, 
        user_id: str,
        project_id: Optional[str] = None,
        start_date:Optional[datetime] = None, 
        end_date:Optional[datetime] = None
    ):
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        # Verify permissions
        
        # Default to last 30 days
        if not start_date:
            start_date = timezone.now() - timedelta(days=30)
        if not end_date:
            end_date = timezone.now()
        
        # Query aggregated metrics
        metrics = UserActivityMetrics.objects.filter(
            user_id=user_id,
            organization=ctx.organization,
            period_start__gte=start_date,
            period_end__lte=end_date
        )
        
        if project_id:
            metrics = metrics.filter(project__project_id=project_id)
        
        # Aggregate across time periods
        summary = metrics.aggregate(
            total_annotations=Sum('annotations_created'),
            manual_annotations=Sum('manual_annotations'),
            ai_edited=Sum('ai_predictions_edited'),
            ai_accepted=Sum('ai_predictions_accepted'),
            images_reviewed=Sum('images_reviewed'),
            images_finalized=Sum('images_finalized'),
            avg_time=Avg('avg_annotation_time_seconds'),
            avg_edit_mag=Avg('avg_edit_magnitude'),
        )
        
        # Get user info
        user = User.objects.get(id=user_id)
        
        # Get session count
        session_count = ActivityEvent.objects.filter(
            user_id=user_id,
            organization=ctx.organization,
            timestamp__gte=start_date,
            timestamp__lte=end_date
        ).values('session_id').distinct().count()
        
        # Get last activity
        last_activity = metrics.aggregate(last=Max('last_activity'))['last']
        
        return UserActivitySummary(
            user_id=str(user.id),
            username=user.username,
            full_name=user.get_full_name() or user.username,
            total_annotations=summary['total_annotations'] or 0,
            manual_annotations=summary['manual_annotations'] or 0,
            ai_predictions_edited=summary['ai_edited'] or 0,
            ai_predictions_accepted=summary['ai_accepted'] or 0,
            images_reviewed=summary['images_reviewed'] or 0,
            images_finalized=summary['images_finalized'] or 0,
            avg_annotation_time_seconds=summary['avg_time'],
            avg_edit_magnitude=summary['avg_edit_mag'],
            last_activity=last_activity,
            total_sessions=session_count
        )
    return await fetch_user_activity_summary(
        ctx, user_id, project_id, start_date, end_date
    )


@router.get("/projects/{project_id}/progress", response_model=ProjectProgressSummary)
async def get_project_progress(
    project_id: UUID,
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get real-time project progress and quality metrics.
    
    **Performance**: Combines pre-aggregated metrics with real-time counts.
    """
    @sync_to_async
    def fetch_project_progress(project_id:UUID):
        from projects.models import Project, ProjectImage
        from annotations.models import Annotation
        
        project = Project.objects.select_related('organization').get(project_id=project_id)
        
        # Get latest daily metrics
        latest_metrics = ProjectActivityMetrics.objects.filter(
            project=project,
            granularity='day'
        ).order_by('-period_start').first()
        
        # If no metrics or stale (>1 hour), compute real-time
        if not latest_metrics or (timezone.now() - latest_metrics.updated_at) > timedelta(hours=1):
            # Real-time counts (cached for 5 minutes recommended)
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
            
            # Prediction quality
            predictions = Annotation.objects.filter(
                project_image__project=project,
                annotation_source='prediction',
                is_active=True
            ).aggregate(
                untouched=Count('id', filter=Q(version=1)),
                edited=Count('id', filter=Q(version__gt=1)),
            )
            
            rejected = Annotation.objects.filter(
                project_image__project=project,
                annotation_source='prediction',
                is_deleted=True
            ).count()
            
            total = status_counts['total']
            finalized = status_counts['finalized']
            completion = (finalized / total * 100) if total > 0 else 0
            
            total_predictions = predictions['untouched'] + predictions['edited'] + rejected
            acceptance = (
                (predictions['untouched'] + predictions['edited']) / total_predictions * 100
                if total_predictions > 0 else 0
            )
            
            return ProjectProgressSummary(
                project_id=str(project.project_id),
                project_name=project.name,
                total_images=total,
                images_unannotated=status_counts['unannotated'],
                images_annotated=status_counts['annotated'],
                images_reviewed=status_counts['reviewed'],
                images_finalized=finalized,
                completion_percentage=round(completion, 2),
                untouched_predictions=predictions['untouched'],
                edited_predictions=predictions['edited'],
                rejected_predictions=rejected,
                prediction_acceptance_rate=round(acceptance, 2),
                annotations_per_hour=None,
                active_contributors=0
            )
        
        # Use cached metrics
        total = latest_metrics.total_images
        completion = (
            (latest_metrics.images_finalized / total * 100) if total > 0 else 0
        )
        
        total_predictions = (
            latest_metrics.untouched_predictions +
            latest_metrics.edited_predictions +
            latest_metrics.rejected_predictions
        )
        acceptance = (
            (latest_metrics.untouched_predictions + latest_metrics.edited_predictions) /
            total_predictions * 100 if total_predictions > 0 else 0
        )
        
        return ProjectProgressSummary(
            project_id=str(project.project_id),
            project_name=project.name,
            total_images=latest_metrics.total_images,
            images_unannotated=latest_metrics.images_unannotated,
            images_annotated=latest_metrics.images_annotated,
            images_reviewed=latest_metrics.images_reviewed,
            images_finalized=latest_metrics.images_finalized,
            completion_percentage=round(completion, 2),
            untouched_predictions=latest_metrics.untouched_predictions,
            edited_predictions=latest_metrics.edited_predictions,
            rejected_predictions=latest_metrics.rejected_predictions,
            prediction_acceptance_rate=round(acceptance, 2),
            annotations_per_hour=float(latest_metrics.annotations_per_hour) if latest_metrics.annotations_per_hour else None,
            active_contributors=latest_metrics.active_users
        )

    return await fetch_project_progress(project_id)

@router.get("/projects/{project_id}/leaderboard", response_model=List[LeaderboardEntry])
async def get_project_leaderboard(
    project_id: UUID,
    metric: str = Query('annotations_created', regex='^(annotations_created|images_reviewed|images_finalized)$'),
    period_days: int = Query(30, ge=1, le=365),
    limit: int = Query(10, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get project leaderboard for specified metric.
    
    **Supported Metrics**:
    - annotations_created
    - images_reviewed
    - images_finalized
    """
    @sync_to_async
    def fetch_project_leaderboard(project_id: UUID, metric:str, period_days:int, limit:int):
        from projects.models import Project
        
        project = Project.objects.get(project_id=project_id)
        
        start_date = timezone.now() - timedelta(days=period_days)
        
        # Query metrics
        leaderboard_data = UserActivityMetrics.objects.filter(
            project=project,
            period_start__gte=start_date
        ).values('user__id', 'user__username', 'user__first_name', 'user__last_name').annotate(
            total=Sum(metric)
        ).order_by('-total')[:limit]
        
        # Calculate total for percentages
        grand_total = sum(entry['total'] for entry in leaderboard_data)
        
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
    
    return await fetch_project_leaderboard(
        project_id, metric, period_days, limit
    )


@router.get("/projects/{project_id}/timeline", response_model=List[ActivityTimelineEvent])
async def get_project_timeline(
    project_id: UUID,
    event_types: Optional[List[str]] = Query(None),
    user_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Get activity timeline for a project.
    
    **Use Case**: Activity feed, audit logs.
    **Performance**: Indexed time-series queries on ActivityEvent.
    """
    @sync_to_async
    def fetch_project_timeline(
        project_id:UUID,
        event_types:Optional[List[str]] = None,
        user_id:Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ):
        from projects.models import Project
        
        project = Project.objects.get(project_id=project_id)
        
        # Build query
        events = ActivityEvent.objects.filter(project=project)
        
        if event_types:
            events = events.filter(event_type__in=event_types)
        
        if user_id:
            events = events.filter(user_id=user_id)
        
        if start_date:
            events = events.filter(timestamp__gte=start_date)
        
        if end_date:
            events = events.filter(timestamp__lte=end_date)
        
        # Paginate and fetch
        events = events.select_related('user').order_by('-timestamp')[offset:offset+limit]
        
        # Format response
        timeline = []
        for event in events:
            timeline.append(ActivityTimelineEvent(
                event_id=str(event.event_id),
                event_type=event.event_type,
                user=event.user.username if event.user else 'System',
                timestamp=event.timestamp,
                project=project.name,
                metadata=event.metadata
            ))
        
        return timeline

    return await fetch_project_timeline(
        project_id=project_id, event_types=event_types,
        user_id=user_id, start_date=start_date, end_date=end_date,
        limit=limit, offset=offset
    )

@router.get("/projects/{project_id}/prediction-quality", response_model=PredictionQualityMetrics)
async def get_prediction_quality_metrics(
    project_id: UUID,
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Analyze quality of AI predictions in a project.
    
    **Key Metrics**:
    - Untouched predictions (accepted as-is)
    - Minor vs major edits
    - Class changes
    - Rejections
    """
    @sync_to_async
    def fetch_prediction_quality_metrics(project_id: UUID):
        from projects.models import Project
        from annotations.models import Annotation
        
        project = Project.objects.get(project_id=project_id)
        
        # Get all predictions
        predictions = Annotation.objects.filter(
            project_image__project=project,
            annotation_source='prediction'
        )
        
        total_predictions = predictions.count()
        
        if total_predictions == 0:
            return PredictionQualityMetrics(
                total_predictions=0,
                untouched=0,
                accepted_without_edit=0,
                minor_edits=0,
                major_edits=0,
                class_changes=0,
                rejected=0,
                untouched_percentage=0,
                acceptance_rate=0,
                avg_edit_magnitude=None
            )
        
        # Categorize predictions
        quality_breakdown = predictions.aggregate(
            untouched=Count('id', filter=Q(version=1, is_active=True, is_deleted=False)),
            minor_edits=Count('id', filter=Q(
                edit_type=Annotation.EditType.MINOR,
                is_active=True
            )),
            major_edits=Count('id', filter=Q(
                edit_type=Annotation.EditType.MAJOR,
                is_active=True
            )),
            class_changes=Count('id', filter=Q(
                edit_type=Annotation.EditType.CLASS_CHANGE,
                is_active=True
            )),
            rejected=Count('id', filter=Q(is_deleted=True)),
            avg_edit=Avg('edit_magnitude', filter=Q(edit_magnitude__isnull=False))
        )
        
        untouched = quality_breakdown['untouched']
        accepted = untouched + quality_breakdown['minor_edits'] + quality_breakdown['major_edits']
        
        return PredictionQualityMetrics(
            total_predictions=total_predictions,
            untouched=untouched,
            accepted_without_edit=untouched,
            minor_edits=quality_breakdown['minor_edits'],
            major_edits=quality_breakdown['major_edits'],
            class_changes=quality_breakdown['class_changes'],
            rejected=quality_breakdown['rejected'],
            untouched_percentage=round((untouched / total_predictions * 100), 2),
            acceptance_rate=round((accepted / total_predictions * 100), 2),
            avg_edit_magnitude=quality_breakdown['avg_edit'] if quality_breakdown['avg_edit'] else None
        )
        
    return await fetch_prediction_quality_metrics(
        project_id=project_id
    )


@router.get("/activity-heatmap")
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
