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

# ============= Version CRUD Endpoints =============

@router.post(
    "/{project_id}/versions",
    response_model=VersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Dataset Version"
)
async def create_version(
    project_id: UUID,
    data: VersionCreate,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Create a new dataset version."""
    
    project_ctx.require_edit_permission()
    
    @sync_to_async
    def create_version_record(project, data, user):
        # Get next version number
        last_version = Version.objects.filter(project=project).order_by('-version_number').first()
        next_version_number = (last_version.version_number + 1) if last_version else 1
        
        # Create version
        version = Version.objects.create(
            project=project,
            version_name=data.version_name,
            version_number=next_version_number,
            description=data.description,
            export_format=data.export_format,
            export_config=data.export_config,
            export_status='pending',
            created_by=user
        )
        
        return version
    
    version = await create_version_record(project_ctx.project, data, project_ctx.user)
    
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