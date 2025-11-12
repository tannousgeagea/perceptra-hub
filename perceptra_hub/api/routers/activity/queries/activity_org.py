from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import List, Optional
from datetime import date
from datetime import datetime, timedelta
from pydantic import BaseModel
from django.utils import timezone
from django.db.models import Sum, Avg, Count, Q, Max
from activity.models import (
    OrgActivitySummary
)

from uuid import UUID
from asgiref.sync import sync_to_async
from api.dependencies import RequestContext, get_request_context

router = APIRouter(prefix="/activity",)

class OrgActivitySummaryResponse(BaseModel):
    date: date
    total_events: int
    active_users: int
    annotation_events: int
    image_events: int
    annotations_created: int
    images_reviewed: int
    images_uploaded: int


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