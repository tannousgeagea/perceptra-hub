"""
Production-ready analytics and statistics endpoints.

Features:
- Optimized queries with select_related/prefetch_related
- Caching with Redis
- Async support
- Proper error handling
- Response models with validation
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Dict, Optional
from uuid import UUID
from pydantic import BaseModel, Field
from datetime import datetime, date, timedelta
import logging
from asgiref.sync import sync_to_async

from django.db.models import Count, Q, Avg, Sum, Max, Min, F
from django.db.models.functions import TruncDate, TruncMonth

from api.dependencies import get_project_context, ProjectContext
from projects.models import Project, ProjectImage
from api.routers.analytics.schemas import ImageStatsResponse
from api.routers.analytics.utils import cache_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Analytics"])

# ============= Image Statistics =============

@router.get(
    "/{project_id}/analytics/images",
    response_model=ImageStatsResponse,
    summary="Get Image Statistics"
)
@cache_response(timeout=300)
async def get_image_stats(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
    days: int = Query(30, ge=1, le=365, description="Days for trend analysis")
):
    """Get detailed image statistics and trends."""
    
    @sync_to_async
    def get_stats(project: Project, days: int):
        images_qs = ProjectImage.objects.filter(
            project=project,
            is_active=True
        ).select_related('image', 'mode')
        
        total = images_qs.count()
        
        # Status breakdown
        status_counts = images_qs.values('status').annotate(
            count=Count('id')
        )
        by_status = {item['status']: item['count'] for item in status_counts}
        
        # Split breakdown
        split_counts = images_qs.filter(
            mode__isnull=False
        ).values('mode__mode').annotate(
            count=Count('id')
        )
        
        by_split = {item['mode__mode']: item['count'] for item in split_counts}
        
        # Job status breakdown
        job_status_counts = images_qs.values('job_assignment_status').annotate(
            count=Count('id')
        )
        by_job_status = {item['job_assignment_status']: item['count'] for item in job_status_counts}
        
        # Upload trend
        cutoff_date = datetime.now() - timedelta(days=days)
        trend_data = images_qs.filter(
            added_at__gte=cutoff_date
        ).annotate(
            date=TruncDate('added_at')
        ).values('date').annotate(
            count=Count('id')
        ).order_by('date')
        
        upload_trend = [
            {
                "date": item['date'].isoformat(),
                "count": item['count']
            }
            for item in trend_data
        ]
        
        # Dimension stats
        dimension_stats = images_qs.aggregate(
            avg_width=Avg('image__width'),
            avg_height=Avg('image__height'),
            avg_file_size=Avg('image__file_size')
        )
        
        return {
            "total": total,
            "by_status": by_status,
            "by_split": by_split,
            "by_job_status": by_job_status,
            "upload_trend": upload_trend,
            "average_dimensions": {
                "width": round(dimension_stats['avg_width'] or 0, 2),
                "height": round(dimension_stats['avg_height'] or 0, 2)
            },
            "average_file_size_mb": round((dimension_stats['avg_file_size'] or 0) / (1024 * 1024), 2)
        }
    
    return await get_stats(project_ctx.project, days)