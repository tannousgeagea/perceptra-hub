"""
FastAPI routes for project and job management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
import logging
from asgiref.sync import sync_to_async

from api.dependencies import get_request_context, RequestContext
from projects.models import (
    Project
)
from django.contrib.auth import get_user_model
from api.routers.projects.schemas import ProjectResponse


User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.get("", summary="List Projects")
async def list_projects(
    ctx: RequestContext = Depends(get_request_context),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    is_active: Optional[bool] = Query(None),
    project_type_id: Optional[int] = Query(None)
):
    """List all projects for organization."""
    
    @sync_to_async
    def get_projects(org, skip, limit, is_active, project_type_id):
        queryset = Project.objects.filter(
            organization=org,
            is_deleted=False
        )
        
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active)
        
        if project_type_id:
            queryset = queryset.filter(project_type_id=project_type_id)
        
        total = queryset.count()
        projects = list(queryset.select_related('project_type', 'visibility')[skip:skip + limit])
        
        return {"total": total, "projects": projects}
    
    result = await get_projects(ctx.organization, skip, limit, is_active, project_type_id)
    
    return {
        "total": result["total"],
        "projects": [
            ProjectResponse(
                id=str(p.id),
                project_id=str(p.project_id),
                name=p.name,
                description=p.description,
                project_type={"id": p.project_type.id, "name": p.project_type.name},
                visibility={"id": p.visibility.id, "name": p.visibility.name},
                is_active=p.is_active,
                is_deleted=p.is_deleted,
                created_at=p.created_at.isoformat(),
                updated_at=p.updated_at.isoformat(),
                last_edited=p.last_edited.isoformat()
            )
            for p in result["projects"]
        ]
    }