
"""
FastAPI routes for project and job management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from uuid import UUID
import uuid
import logging
from asgiref.sync import sync_to_async

from api.dependencies import ProjectContext, get_project_context
from projects.models import (
    Project,  ProjectImage, 
    ImageMode
)
from jobs.models import Job, JobImage
from images.models import Image
from django.db import transaction
from django.contrib.auth import get_user_model
from api.routers.projects.schemas import AddImagesToProjectRequest
from common_utils.jobs.utils import assign_uploaded_image_to_batch

User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Projects"])



@router.post(
    "/{project_id}/images",
    status_code=status.HTTP_201_CREATED,
    summary="Add Images to Project"
)
async def add_images_to_project(
    project_id: UUID,
    data: AddImagesToProjectRequest,
    project_ctx:ProjectContext = Depends(get_project_context),
):
    """Add images to project and optionally assign to job."""
    
    @sync_to_async
    def add_images(org, project, data, user):
                
        # Get images
        images = Image.objects.filter(
            organization=org,
            image_id__in=data.image_ids
        )
        
        if not images.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No valid images found"
            )
        
        # Get mode if specified
        mode = None
        if data.mode_id:
            try:
                mode = ImageMode.objects.get(id=data.mode_id)
            except ImageMode.DoesNotExist:
                pass
        
        with transaction.atomic():
            created_links = []
            for image in images:
                
                # Check if already linked
                if ProjectImage.objects.filter(project=project, image=image).exists():
                    continue
                
                project_image = ProjectImage(
                    project=project,
                    image=image,
                    mode=mode,
                    priority=data.priority,
                    added_by=user,
                    status='unannotated',
                    job_assignment_status='waiting'
                )
                created_links.append(project_image)
            
            # Bulk create all project images
            ProjectImage.objects.bulk_create(created_links)
            
            # Generate batch ID for this upload batch
            batch_id = str(uuid.uuid4()) if data.auto_assign_job else None
            assigned_jobs = set()
            
            # Assign to jobs
            for project_image in created_links:
                assigned_job = assign_uploaded_image_to_batch(
                    project_image=project_image,
                    batch_id=batch_id,
                    user=user if data.auto_assign_job else None
                )
                if assigned_job:
                    assigned_jobs.add(assigned_job.id)
        
        return {
            "added": len(created_links),
            "image_ids": [str(img.image.image_id) for img in created_links],
            "batch_id": batch_id,
            "job_assigned": list(assigned_jobs)
        }
    
    result = await add_images(project_ctx.organization, project_ctx.project, data, project_ctx.user)
    
    return {
        "message": f"Added {result['added']} images to project",
        "added_count": result["added"],
        "image_ids": result["image_ids"],
        "batch_id": result["batch_id"],
        "jobs_assigned": result["jobs_assigned"]
    }