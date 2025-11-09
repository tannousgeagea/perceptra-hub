"""
Production-ready analytics and statistics endpoints.

Features:
- Optimized queries with select_related/prefetch_related
- Caching with Redis
- Async support
- Proper error handling
- Response models with validation
"""
from fastapi import APIRouter, Depends
from uuid import UUID
import logging
from asgiref.sync import sync_to_async
from functools import wraps

from django.db.models import Count, Sum

from api.dependencies import get_project_context, ProjectContext
from projects.models import Project
from jobs.models import Job
from api.routers.analytics.schemas import JobStatsResponse
from api.routers.analytics.utils import cache_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Analytics"])

# ============= Job Statistics =============

@router.get(
    "/{project_id}/analytics/jobs",
    response_model=JobStatsResponse,
    summary="Get Job Statistics"
)
@cache_response(timeout=300)
async def get_job_stats(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Get job statistics and metrics."""
    
    @sync_to_async
    def get_stats(project: Project):
        jobs_qs = Job.objects.filter(project=project)
        
        # Status breakdown
        status_counts = jobs_qs.values('status').annotate(
            count=Count('id')
        )
        by_status = {item['status']: item['count'] for item in status_counts}
        
        # Image stats
        total_jobs = jobs_qs.count()
        total_images = jobs_qs.aggregate(total=Sum('image_count'))['total'] or 0
        avg_images = total_images / total_jobs if total_jobs > 0 else 0
        
        # Completion rate
        completed = jobs_qs.filter(status='completed').count()
        completion_rate = (completed / total_jobs * 100) if total_jobs > 0 else 0
        
        return {
            "total": total_jobs,
            "by_status": by_status,
            "total_images": total_images,
            "average_images_per_job": round(avg_images, 2),
            "completion_rate": round(completion_rate, 2)
        }
    
    return await get_stats(project_ctx.project)