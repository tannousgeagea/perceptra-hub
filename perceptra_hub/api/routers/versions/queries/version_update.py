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
from api.routers.versions.schemas import VersionUpdate, VersionResponse
from projects.models import Project, ProjectImage, Version, VersionImage


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")




@router.put(
    "/{project_id}/versions/{version_id}",
    response_model=VersionResponse,
    summary="Update Dataset Version"
)
async def update_version(
    project_id: UUID,
    version_id: UUID,
    data: VersionUpdate,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Update dataset version details."""
    
    project_ctx.require_edit_permission()
    
    @sync_to_async
    def update_version_record(project, version_id, data):
        try:
            version = Version.objects.get(version_id=version_id, project=project)
        except Version.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
        
        # Can't update if export is processing or completed
        if version.export_status in ['processing', 'completed']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot update version with status '{version.export_status}'"
            )
        
        if data.version_name is not None:
            version.version_name = data.version_name
        
        if data.description is not None:
            version.description = data.description
        
        if data.export_config is not None:
            version.export_config = data.export_config
        
        version.save()
        
        return version, version.created_by
    
    version, created_by = await update_version_record(project_ctx.project, version_id, data)
    
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
        created_by=created_by.username if created_by else None
    )