
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
    "/{project_id}/versions/{version_id}",
    response_model=VersionResponse,
    summary="Get Dataset Version"
)
async def get_version(
    project_id: UUID,
    version_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Get dataset version details."""
    
    @sync_to_async
    def fetch_version(project, version_id):
        try:
            return Version.objects.select_related('created_by').get(
                version_id=version_id,
                project=project
            )
        except Version.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
    
    version = await fetch_version(project_ctx.project, version_id)
    
    return VersionResponse(
        id=str(version.id),
        version_id=str(version.version_id),
        version_name=version.version_name,
        version_number=version.version_number,
        description=version.description,
        export_format=version.export_format,
        export_status=version.export_status,
        total_images=version.total_images,
        total_annotations=version.total_annotations,
        train_count=version.train_count,
        val_count=version.val_count,
        test_count=version.test_count,
        file_size=version.file_size,
        is_ready=version.is_ready,
        created_at=version.created_at.isoformat(),
        exported_at=version.exported_at.isoformat() if version.exported_at else None,
        created_by=version.created_by.username if version.created_by else None
    )


