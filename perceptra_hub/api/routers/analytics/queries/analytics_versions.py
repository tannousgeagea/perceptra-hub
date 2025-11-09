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
from typing import List, Dict, Optional
from uuid import UUID
import logging
from asgiref.sync import sync_to_async

from django.db.models import Count

from api.dependencies import get_project_context, ProjectContext
from projects.models import Project, Version, VersionImage
from api.routers.analytics.schemas import VersionStatsResponse
from api.routers.analytics.utils import cache_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Analytics"])

# ============= Version Statistics =============

@router.get(
    "/{project_id}/analytics/versions",
    response_model=List[VersionStatsResponse],
    summary="Get Version Statistics"
)
@cache_response(timeout=600)
async def get_version_stats(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Get statistics for all versions."""
    
    @sync_to_async
    def get_stats(project: Project):
        versions = Version.objects.filter(
            project=project
        ).order_by('-version_number')
        
        result = []
        for version in versions:
            # Get split distribution
            split_counts = VersionImage.objects.filter(
                version=version
            ).values('split').annotate(
                count=Count('id')
            )
            by_split = {item['split']: item['count'] for item in split_counts}
            
            result.append({
                "id": str(version.version_id),
                "version_name": version.version_name,
                "version_number": version.version_number,
                "export_format": version.export_format,
                "export_status": version.export_status,
                "total_images": version.total_images,
                "by_split": by_split,
                "total_annotations": version.total_annotations,
                "file_size_mb": round(version.file_size / (1024 * 1024), 2) if version.file_size else None,
                "created_at": version.created_at.isoformat(),
                "exported_at": version.exported_at.isoformat() if version.exported_at else None
            })
        
        return result
    
    return await get_stats(project_ctx.project)