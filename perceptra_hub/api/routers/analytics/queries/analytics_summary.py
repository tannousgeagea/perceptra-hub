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
from projects.models import Project, ProjectImage, Version
from annotations.models import Annotation
from jobs.models import Job
from api.routers.analytics.schemas import ProjectSummaryResponse
from api.routers.analytics.utils import cache_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Analytics"])

# ============= Project Summary =============

@router.get(
    "/{project_id}/analytics/summary",
    response_model=ProjectSummaryResponse,
    summary="Get Project Summary"
)
@cache_response(timeout=300)
async def get_project_summary(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Get comprehensive project summary with all key metrics."""
    
    @sync_to_async
    def get_summary(project: Project):
        # Image stats
        images_qs = ProjectImage.objects.filter(
            project=project,
            is_active=True
        )
        
        image_stats = images_qs.aggregate(
            total=Count('id'),
            annotated=Count('id', filter=Q(annotated=True)),
            reviewed=Count('id', filter=Q(reviewed=True)),
            finalized=Count('id', filter=Q(finalized=True)),
            null_marked=Count('id', filter=Q(marked_as_null=True))
        )
        
        # Annotation stats
        annotations_qs = Annotation.objects.filter(
            project_image__project=project,
            is_active=True
        )
        
        annotation_stats = annotations_qs.aggregate(
            total=Count('id'),
            manual=Count('id', filter=Q(annotation_source='manual')),
            prediction=Count('id', filter=Q(annotation_source='prediction'))
        )
        
        # Job stats
        jobs_qs = Job.objects.filter(project=project)
        job_stats = jobs_qs.aggregate(
            total=Count('id'),
            active=Count('id', filter=Q(status__in=['unassigned', 'assigned', 'in_review'])),
            completed=Count('id', filter=Q(status='completed'))
        )
        
        # Version stats
        versions = Version.objects.filter(project=project)
        latest_version = versions.order_by('-version_number').first()
        
        return {
            "project_id": str(project.project_id),
            "project_name": project.name,
            "description": project.description,
            "project_type": project.project_type.name,
            "visibility": project.visibility.name,
            "created_at": project.created_at.isoformat(),
            "last_edited": project.last_edited.isoformat(),
            
            "total_images": image_stats['total'] or 0,
            "annotated_images": image_stats['annotated'] or 0,
            "reviewed_images": image_stats['reviewed'] or 0,
            "finalized_images": image_stats['finalized'] or 0,
            "null_images": image_stats['null_marked'] or 0,
            
            "total_annotations": annotation_stats['total'] or 0,
            "manual_annotations": annotation_stats['manual'] or 0,
            "prediction_annotations": annotation_stats['prediction'] or 0,
            
            "total_jobs": job_stats['total'] or 0,
            "active_jobs": job_stats['active'] or 0,
            "completed_jobs": job_stats['completed'] or 0,
            
            "total_versions": versions.count(),
            "latest_version": f"v{latest_version.version_number}" if latest_version else None
        }
    
    return await get_summary(project_ctx.project)
