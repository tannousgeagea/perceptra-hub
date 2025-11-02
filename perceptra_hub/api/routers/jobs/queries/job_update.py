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
from api.routers.jobs.schemas import JobResponse, JobUpdate

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

@router.put(
    "/{project_id}/jobs/{job_id}",
    response_model=JobResponse,
    summary="Update Job",
    description="Update job details (name, description, status)"
)
async def update_job(
    project_id: UUID,
    job_id: int,
    data: JobUpdate,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Update job details."""
    
    @sync_to_async
    def update_job_record(project, job_id, data, user=None):        
        # Get job
        try:
            job = Job.objects.select_related('assignee').get(
                id=job_id,
                project=project
            )
        except Job.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        # Update fields if provided
        if data.name is not None:
            job.name = data.name
        
        if data.description is not None:
            job.description = data.description
        
        if data.status is not None:
            # Validate status choice
            valid_statuses = ['unassigned', 'assigned', 'in_review', 'completed', 'sliced']
            if data.status not in valid_statuses:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
                )
                
            # Validate completion requirements
            if data.status == "completed":
                from jobs.models import JobImage
                
                job_image_ids = JobImage.objects.filter(job=job).values_list("project_image_id", flat=True)
                unreviewed_count = ProjectImage.objects.filter(
                    id__in=job_image_ids, 
                    reviewed=False
                ).count()
                
                if unreviewed_count > 0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Cannot mark as completed. {unreviewed_count} images are not reviewed."
                    )
                    
            job.status = data.status

        
        if data.assignee_id is not None:
            # Handle assignee change
            if data.assignee_id == "":
                # Empty string means unassign
                job.assignee = None
            else:
                try:
                    assignee = User.objects.get(id=data.assignee_id)
                    job.assignee = assignee
                except User.DoesNotExist:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Assignee user not found"
                    )
        
        job.updated_by = user
        job.save()
        
        return job
    
    job = await update_job_record(project_ctx.project, job_id, data, user=project_ctx.user)
    
    return JobResponse(
        id=str(job.id),
        name=job.name,
        description=job.description,
        status=job.status,
        image_count=job.image_count,
        assignee={
            "id": str(job.assignee.id),
            "username": job.assignee.username,
            "email": job.assignee.email
        } if job.assignee else None,
        batch_id=job.batch_id,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat()
    )