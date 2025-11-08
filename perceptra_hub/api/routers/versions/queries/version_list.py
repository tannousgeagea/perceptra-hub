
"""
FastAPI routes for dataset version management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field
import logging
from asgiref.sync import sync_to_async
from datetime import datetime

from api.dependencies import get_project_context, ProjectContext
from api.routers.versions.schemas import VersionCreate, VersionResponse
from projects.models import Project, ProjectImage, Version, VersionImage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")

@router.get(
    "/{project_id}/versions",
    summary="List Dataset Versions"
)
async def list_versions(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    export_status: Optional[str] = Query(None)
):
    """List all dataset versions for project."""
    
    @sync_to_async
    def get_versions(project, skip, limit, export_status):
        queryset = Version.objects.filter(project=project)
        
        if export_status:
            queryset = queryset.filter(export_status=export_status)
        
        total = queryset.count()
        versions = list(
            queryset.select_related('created_by')[skip:skip + limit]
        )
        
        return {"total": total, "versions": versions}
    
    result = await get_versions(project_ctx.project, skip, limit, export_status)
    
    return {
        "total": result["total"],
        "versions": [
            VersionResponse(
                id=str(v.id),
                version_id=str(v.version_id),
                version_name=v.version_name,
                version_number=v.version_number,
                description=v.description,
                export_format=v.export_format,
                export_status=v.export_status,
                total_images=v.total_images,
                total_annotations=v.total_annotations,
                train_count=v.train_count,
                val_count=v.val_count,
                test_count=v.test_count,
                file_size=v.file_size,
                is_ready=v.is_ready,
                created_at=v.created_at.isoformat(),
                exported_at=v.exported_at.isoformat() if v.exported_at else None,
                created_by=v.created_by.username if v.created_by else None
            )
            for v in result["versions"]
        ]
    }