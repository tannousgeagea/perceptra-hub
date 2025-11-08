

"""
FastAPI routes for dataset version management.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from uuid import UUID
import logging
from asgiref.sync import sync_to_async
from datetime import datetime

from api.routers.versions.schemas import ExportConfigRequest
from api.dependencies import get_project_context, ProjectContext
from projects.models import Version

from fastapi import BackgroundTasks
from typing import Optional
import threading


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")


@router.post(
    "/{project_id}/versions/{version_id}/export",
    summary="Export Dataset",
    description="Trigger async dataset export with augmentation support"
)
async def export_dataset(
    project_id: UUID,
    version_id: UUID,
    config: Optional[ExportConfigRequest] = None,
    background_tasks: BackgroundTasks = None,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """
    Trigger dataset export (async background task).
    
    Supports:
    - Multiple formats: YOLO, COCO, Pascal VOC
    - Image resizing and quality control
    - Data augmentation with albumentations
    - Automatic zip packaging
    """
    
    project_ctx.require_edit_permission()
    
    @sync_to_async
    def prepare_export(project, version_id, export_config):
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
        
        # Update export config if provided
        if export_config:
            version.export_config = export_config.dict()
        
        # Update status to processing
        version.export_status = 'processing'
        version.save()
        
        return version.id
    
    config = config or ExportConfigRequest()
    version_pk = await prepare_export(project_ctx.project, version_id, config)
    
    # Run export in background thread (or use Celery in production)
    def run_export():
        from common_utils.dataset_export.manager import DatasetExportManager
        DatasetExportManager.export_version(version_pk)
    
    thread = threading.Thread(target=run_export)
    thread.start()
    
    
    # Alternative: Use Celery for production
    # from dataset_export import export_dataset_task
    # export_dataset_task.delay(version_pk)
    
    return {
        "message": "Dataset export started",
        "version_id": str(version_id),
        "status": "processing",
        "config": config.dict(),
        "estimated_time": "5-30 minutes depending on dataset size"
    }


@router.get(
    "/{project_id}/versions/{version_id}/export-status",
    summary="Check Export Status",
    description="Get current export status and progress"
)
async def get_export_status(
    project_id: UUID,
    version_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Check export status."""
    
    @sync_to_async
    def get_status(project, version_id):
        try:
            version = Version.objects.get(version_id=version_id, project=project)
        except Version.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
        
        return {
            "version_id": str(version.version_id),
            "version_name": version.version_name,
            "export_status": version.export_status,
            "export_format": version.export_format,
            "is_ready": version.is_ready,
            "file_size": version.file_size,
            "file_size_mb": round(version.file_size / (1024 * 1024), 2) if version.file_size else None,
            "exported_at": version.exported_at.isoformat() if version.exported_at else None,
            "error_log": version.error_log,
            "total_images": version.total_images,
            "download_count": version.download_count
        }
    
    return await get_status(project_ctx.project, version_id)


@router.post(
    "/{project_id}/versions/{version_id}/retry-export",
    summary="Retry Failed Export",
    description="Retry a failed export"
)
async def retry_export(
    project_id: UUID,
    version_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Retry a failed export."""
    
    project_ctx.require_edit_permission()
    
    @sync_to_async
    def prepare_retry(project, version_id):
        try:
            version = Version.objects.get(version_id=version_id, project=project)
        except Version.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
        
        if version.export_status != 'failed':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot retry export with status '{version.export_status}'"
            )
        
        # Reset status
        version.export_status = 'processing'
        version.error_log = None
        version.save()
        
        return version.id
    
    version_pk = await prepare_retry(project_ctx.project, version_id)
    
    # Run export in background
    def run_export():
        from common_utils.dataset_export.manager import DatasetExportManager
        DatasetExportManager.export_version(version_pk)
    
    thread = threading.Thread(target=run_export)
    thread.start()
    
    return {
        "message": "Export retry started",
        "version_id": str(version_id),
        "status": "processing"
    }