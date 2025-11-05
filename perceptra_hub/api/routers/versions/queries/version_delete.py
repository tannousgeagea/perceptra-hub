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

@router.delete(
    "/{project_id}/versions/{version_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Dataset Version"
)
async def delete_version(
    project_id: UUID,
    version_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Delete a dataset version."""
    
    project_ctx.require_edit_permission()
    
    @sync_to_async
    def delete_version_record(project, version_id):
        try:
            version = Version.objects.get(version_id=version_id, project=project)
        except Version.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
        
        # Delete dataset file if exists
        if version.dataset_file:
            version.dataset_file.delete()
        
        version.delete()
    
    await delete_version_record(project_ctx.project, version_id)

