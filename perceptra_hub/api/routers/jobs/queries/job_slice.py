# api/jobs.py


"""
FastAPI routes for project and job management.
"""
import time
from fastapi.routing import APIRoute
from fastapi import Request, Response, Body
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
import logging
from uuid import UUID
from asgiref.sync import sync_to_async
from django.db.models import Count, Q
from api.dependencies import get_project_context, ProjectContext

from django.contrib.auth import get_user_model
from jobs.models import Job, JobImage
from projects.models import ProjectImage
from api.routers.jobs.schemas import JobSplitRequest

User = get_user_model()
logger = logging.getLogger(__name__)

class TimedRoute(APIRoute):
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request):
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            return response

        return custom_route_handler

router = APIRouter(
    prefix="/projects",
    route_class=TimedRoute
)

@router.post(
    "/{project_id}/jobs/{job_id}/split",
    summary="Split Job into Slices",
    description="Split a job into multiple smaller jobs with optional user assignments"
)
async def split_job_into_slices(
    project_id: UUID,
    job_id: int,
    data: JobSplitRequest,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """
    Split job into multiple slices.
    
    - Creates new jobs with distributed images
    - Assigns users if provided
    - Marks original job as 'sliced'
    """
    
    @sync_to_async
    def split_job(project, job_id, data, user=None):
        from jobs.models import Job, JobImage
        from django.db import transaction
        
        # Get job
        try:
            job = Job.objects.select_related('project').get(
                id=job_id,
                project=project
            )
        except Job.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        # Get job images
        job_images = list(JobImage.objects.filter(job=job).select_related('project_image'))
        total_images = len(job_images)
        
        if total_images == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job has no images to split"
            )
        
        if data.number_of_slices > total_images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot split into more slices ({data.number_of_slices}) than images ({total_images})"
            )
        
        if len(data.user_assignments) != data.number_of_slices:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Number of user assignments ({len(data.user_assignments)}) must match number of slices ({data.number_of_slices})"
            )
        
        # Validate all user IDs exist
        user_ids = [uid for uid in data.user_assignments if uid]
        if user_ids:
            existing_users = User.objects.filter(id__in=user_ids).values_list('id', flat=True)
            missing_users = set(user_ids) - set(str(uid) for uid in existing_users)
            if missing_users:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Users not found: {', '.join(missing_users)}"
                )
        
        # Calculate images per slice
        images_per_slice = total_images // data.number_of_slices
        remainder = total_images % data.number_of_slices
        
        created_slices = []
        
        with transaction.atomic():
            slice_start = 0
            
            for i in range(data.number_of_slices):
                # Calculate slice size (distribute remainder across first slices)
                slice_image_count = images_per_slice + (1 if i < remainder else 0)
                slice_images = job_images[slice_start:slice_start + slice_image_count]
                slice_start += slice_image_count
                
                # Get assignee
                assignee_id = data.user_assignments[i]
                assignee = User.objects.get(id=assignee_id) if assignee_id else None
                
                # Create slice job
                job_slice = Job.objects.create(
                    project=job.project,
                    name=f"{job.name} (Slice {i + 1})",
                    description=job.description,
                    status="assigned" if assignee else "unassigned",
                    assignee=assignee,
                    image_count=slice_image_count,
                    batch_id=job.batch_id,
                    created_by=user,
                    updated_by=user,
                )
                
                # Assign images to slice
                JobImage.objects.bulk_create([
                    JobImage(job=job_slice, project_image=ji.project_image)
                    for ji in slice_images
                ])
                
                # Update project images job assignment status
                project_image_ids = [ji.project_image.id for ji in slice_images]
                ProjectImage.objects.filter(id__in=project_image_ids).update(
                    job_assignment_status='assigned'
                )
                
                created_slices.append({
                    "id": str(job_slice.id),
                    "name": job_slice.name,
                    "image_count": slice_image_count,
                    "assignee": {
                        "id": str(assignee.id),
                        "username": assignee.username
                    } if assignee else None
                })
            
            # Mark original job as sliced
            job.status = "sliced"
            job.updated_by = user
            job.save(update_fields=["status", "updated_at"])
        
        return {
            "original_job_id": str(job.id),
            "slices": created_slices
        }
    
    result = await split_job(project_ctx.project, job_id, data, user=project_ctx.user)
    
    return {
        "message": f"Job split into {data.number_of_slices} slices successfully",
        "original_job_id": result["original_job_id"],
        "slices_created": len(result["slices"]),
        "slices": result["slices"]
    }