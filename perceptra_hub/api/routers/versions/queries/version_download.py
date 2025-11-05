

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
from api.routers.versions.schemas import VersionImageAdd, VersionResponse
from projects.models import Project, ProjectImage, Version, VersionImage


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")



@router.get(
    "/{project_id}/versions/{version_id}/download",
    summary="Download Dataset",
    description="Download exported dataset file"
)
async def download_dataset(
    project_id: UUID,
    version_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Get download URL for exported dataset."""
    
    @sync_to_async
    def get_download_info(project, version_id):
        try:
            version = Version.objects.get(version_id=version_id, project=project)
        except Version.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
        
        if not version.is_ready:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset not ready. Status: {version.export_status}"
            )
        
        # Increment download count
        version.download_count += 1
        version.save(update_fields=['download_count'])
        
        # Generate presigned URL if using cloud storage
        if version.storage_key:
            # TODO: Generate presigned URL from storage adapter
            download_url = f"/datasets/{version.storage_key}"
        else:
            download_url = version.dataset_file.url if version.dataset_file else None
        
        return {
            "download_url": download_url,
            "file_size": version.file_size,
            "format": version.export_format
        }
    
    return await get_download_info(project_ctx.project, version_id)