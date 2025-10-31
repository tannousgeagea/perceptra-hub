

"""
FastAPI routes for project and job management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field
import logging
from asgiref.sync import sync_to_async

from api.dependencies import get_project_context, ProjectContext
from jobs.models import Job, JobImage
from images.models import Image

from django.contrib.auth import get_user_model

User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects")


@sync_to_async
def get_dowload_url(image:Image, expiration:int=3600):
    return image.get_download_url(expiration=expiration)


@router.get("/{project_id}/jobs/{job_id}/images", summary="List Job Images")
async def list_job_images(
    project_id: UUID,
    job_id: int,
    project_ctx: ProjectContext = Depends(get_project_context),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    annotated: Optional[bool] = Query(None)
):
    """List images assigned to a job."""
    
    @sync_to_async
    def get_job_images(project, job_id, skip, limit, status_filter, annotated):
        # Verify job belongs to project
        try:
            job = Job.objects.get(id=job_id, project=project)
        except Job.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        # Get job images
        queryset = JobImage.objects.filter(
            job=job,
            project_image__is_active=True
        ).select_related('project_image')
        
        annotated_count = queryset.filter(project_image__status='annotated').count()
        reviewed_count = queryset.filter(project_image__status='reviewed').count()
        unannotated_count = queryset.filter(project_image__status='unannotated').count()
        
        
        # Apply filters on project_image
        if status_filter:
            queryset = queryset.filter(project_image__status=status_filter)
        
        if annotated is not None:
            queryset = queryset.filter(project_image__annotated=annotated)
        
        total = queryset.count()
        
        job_images = list(
            queryset.select_related(
                'project_image__image',
                'project_image__mode'
            ).prefetch_related(
                'project_image__annotations__annotation_class',
                'project_image__annotations__annotation_type'
            )[skip:skip + limit]
        )
        
        return {
            "total": total,
            "annotated": annotated_count,
            "unannotated": unannotated_count,
            "reviewed": reviewed_count,
            "job": {
                "id": str(job.id),
                "name": job.name,
                "status": job.status,
                "assignee": job.assignee.username if job.assignee else None
            },
            "images": [
                {
                    "id": str(ji.project_image.id),
                    "image_id": str(ji.project_image.image.image_id),
                    "name": ji.project_image.image.name,
                    "width": ji.project_image.image.width,
                    "height": ji.project_image.image.height,
                    "storage_key": ji.project_image.image.storage_key,
                    "download_url": ji.project_image.image.get_download_url(expiration=3600),
                    "status": ji.project_image.status,
                    "annotated": ji.project_image.annotated,
                    "reviewed": ji.project_image.reviewed,
                    "priority": ji.project_image.priority,
                    "added_at": ji.created_at.isoformat(),
                    "annotations": [
                        {
                            "id": str(ann.id),
                            "annotation_uid": ann.annotation_uid,
                            "type": ann.annotation_type.name if ann.annotation_type else None,
                            "class_id": ann.annotation_class.class_id,
                            "class_name": ann.annotation_class.name,
                            "color": ann.annotation_class.color,
                            "data": ann.data,
                            "source": ann.annotation_source,
                            "confidence": ann.confidence,
                            "reviewed": ann.reviewed,
                            "is_active": ann.is_active
                        }
                        for ann in ji.project_image.annotations.filter(is_active=True).select_related(
                            'annotation_class', 'annotation_type'
                        )
                    ]
                }
                for ji in job_images
            ]
        }
    
    return await get_job_images(
        project_ctx.project, job_id, skip, limit, status, annotated
    )