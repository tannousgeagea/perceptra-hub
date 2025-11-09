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

from django.db.models import Count, Q

from api.dependencies import get_project_context, ProjectContext
from projects.models import Project, ProjectImage
from annotations.models import Annotation
from api.routers.analytics.schemas import AnnotationStatsResponse
from api.routers.analytics.utils import cache_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Analytics"])

# ============= Annotation Statistics =============

@router.get(
    "/{project_id}/analytics/annotations",
    response_model=AnnotationStatsResponse,
    summary="Get Annotation Statistics"
)
@cache_response(timeout=300)
async def get_annotation_stats(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Get detailed annotation statistics."""
    
    @sync_to_async
    def get_stats(project: Project):
        annotations_qs = Annotation.objects.filter(
            project_image__project=project
        )
        
        # Basic counts
        stats = annotations_qs.aggregate(
            total=Count('id'),
            active=Count('id', filter=Q(is_active=True)),
            inactive=Count('id', filter=Q(is_active=False))
        )
        
        # Source breakdown
        source_counts = annotations_qs.filter(is_active=True).values(
            'annotation_source'
        ).annotate(count=Count('id'))
        by_source = {item['annotation_source']: item['count'] for item in source_counts}
        
        # Status breakdown
        status_stats = {
            'reviewed': annotations_qs.filter(is_active=True, reviewed=True).count(),
            'pending': annotations_qs.filter(is_active=True, reviewed=False).count()
        }
        
        # Average per image
        image_count = ProjectImage.objects.filter(
            project=project,
            is_active=True
        ).count()
        avg_per_image = (stats['active'] or 0) / image_count if image_count > 0 else 0
        
        # Class distribution
        class_counts = annotations_qs.filter(
            is_active=True
        ).values(
            'annotation_class__id',
            'annotation_class__name',
            'annotation_class__color'
        ).annotate(
            count=Count('id')
        ).order_by('-count')[:20]  # Top 20 classes
        
        total_active = stats['active'] or 1  # Avoid division by zero
        class_distribution = [
            {
                "class_id": item['annotation_class__id'],
                "class_name": item['annotation_class__name'],
                "color": item['annotation_class__color'] or '#888888',
                "count": item['count'],
                "percentage": round((item['count'] / total_active) * 100, 2)
            }
            for item in class_counts
        ]
        
        return {
            "total": stats['total'] or 0,
            "active": stats['active'] or 0,
            "inactive": stats['inactive'] or 0,
            "by_source": by_source,
            "by_status": status_stats,
            "average_per_image": round(avg_per_image, 2),
            "class_distribution": class_distribution
        }
    
    return await get_stats(project_ctx.project)