

"""
FastAPI routes for dataset version management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from typing import Optional, List
from uuid import UUID
import logging
from asgiref.sync import sync_to_async
from datetime import datetime

from api.dependencies import get_project_context, ProjectContext
from projects.models import Version


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")


@router.post(
    "/{project_id}/versions/{version_id}/export",
    summary="Export Dataset",
    description="Trigger async dataset export job"
)
async def export_dataset(
    project_id: UUID,
    version_id: UUID,
    background_tasks: BackgroundTasks,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Trigger dataset export (async background task)."""
    
    project_ctx.require_edit_permission()
    
    @sync_to_async
    def prepare_export(project, version_id):
        try:
            version = Version.objects.get(version_id=version_id, project=project)
        except Version.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
        
        if version.total_images == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot export version with no images"
            )
        
        if version.export_status == 'processing':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Export already in progress"
            )
        
        # Update status to processing
        version.export_status = 'processing'
        version.save(update_fields=['export_status'])
        
        return version.id
    
    version_pk = await prepare_export(project_ctx.project, version_id)
    
    # TODO: Add actual background task
    # background_tasks.add_task(export_dataset_task, version_pk)
    
    return {
        "message": "Dataset export started",
        "version_id": str(version_id),
        "status": "processing"
    }