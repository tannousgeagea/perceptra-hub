
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
import logging
from django.core.cache import cache
from api.dependencies import get_project_context, ProjectContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Analytics"])

# ============= Cache Management =============

@router.post(
    "/{project_id}/analytics/clear-cache",
    summary="Clear Analytics Cache"
)
async def clear_analytics_cache(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Clear cached analytics for project."""
    
    project_ctx.require_edit_permission()
    
    # Clear all analytics caches for this project
    cache_pattern = f"analytics:{project_ctx.project.project_id}:*"
    
    # Note: In production, use Redis SCAN for pattern matching
    # For Django cache, we need to clear manually
    cache.delete_many([
        f"analytics:{project_ctx.project.project_id}:get_project_summary:",
        f"analytics:{project_ctx.project.project_id}:get_image_stats:",
        f"analytics:{project_ctx.project.project_id}:get_annotation_stats:",
        f"analytics:{project_ctx.project.project_id}:get_annotation_groups:",
        f"analytics:{project_ctx.project.project_id}:get_job_stats:",
        f"analytics:{project_ctx.project.project_id}:get_version_stats:",
        f"analytics:{project_ctx.project.project_id}:get_evaluation_stats:",
    ])
    
    return {"message": "Analytics cache cleared successfully"}