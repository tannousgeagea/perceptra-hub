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
from asgiref.sync import sync_to_async
from django.db.models import Count

from api.dependencies import get_project_context, ProjectContext
from projects.models import Project
from annotations.models import Annotation, AnnotationGroup
from api.routers.analytics.utils import cache_response
from api.routers.analytics.schemas import AnnotationGroupResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Analytics"])

# ============= Annotation Groups =============

@router.get(
    "/{project_id}/analytics/annotation-groups",
    response_model=List[AnnotationGroupResponse],
    summary="Get Annotation Groups"
)
@cache_response(timeout=600)
async def get_annotation_groups(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Get annotation groups with class distribution."""
    
    @sync_to_async
    def get_groups(project: Project):
        groups = AnnotationGroup.objects.filter(
            project=project
        ).prefetch_related('classes')
        
        # Get annotation counts per class
        class_counts = Annotation.objects.filter(
            project_image__project=project,
            is_active=True
        ).values('annotation_class').annotate(
            count=Count('id')
        )
        class_count_map = {item['annotation_class']: item['count'] for item in class_counts}
        
        result = []
        for group in groups:
            classes = []
            total_group_annotations = 0
            
            for cls in group.classes.all():
                count = class_count_map.get(cls.id, 0)
                total_group_annotations += count
                
                classes.append({
                    "class_id": cls.class_id,
                    "class_name": cls.name,
                    "color": cls.color or '#888888',
                    "count": count,
                    "percentage": 0  # Will calculate after we have total
                })
            
            # Calculate percentages
            if total_group_annotations > 0:
                for cls in classes:
                    cls["percentage"] = round((cls["count"] / total_group_annotations) * 100, 2)
            
            result.append({
                "id": group.id,
                "name": group.name,
                "description": group.description,
                "classes": classes,
                "total_annotations": total_group_annotations
            })
        
        return result
    
    return await get_groups(project_ctx.project)