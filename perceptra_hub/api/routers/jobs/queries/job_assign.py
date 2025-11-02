
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
from api.routers.jobs.schemas import JobResponse

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

@router.patch(
    "/{project_id}/jobs/{job_id}/assign",
    response_model=JobResponse,
    summary="Assign Job to User",
    description="Assign or reassign a job to a user"
)
async def assign_job_to_user(
    project_id: UUID,
    job_id: int,
    assignee_id: int = Body(..., embed=True, description="User ID to assign job to"),
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Assign job to a user."""
    
    @sync_to_async
    def assign_job(project, job_id, assignee_id, user=None):
        from jobs.models import Job
        
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
        
        # Get assignee user
        try:
            assignee = User.objects.get(id=assignee_id)
        except User.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Assign job
        job.assignee = assignee
        job.updated_by = user
        
        # Update status if currently unassigned
        if job.status == 'unassigned':
            job.status = 'assigned'
        
        job.save(update_fields=['assignee', 'status', 'updated_at', 'updated_by'])
        
        return job
    
    job = await assign_job(project_ctx.project, job_id, assignee_id, user=project_ctx.user)
    
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


@router.patch(
    "/{project_id}/jobs/{job_id}/unassign",
    response_model=JobResponse,
    summary="Unassign Job",
    description="Remove assignee from job"
)
async def unassign_job(
    project_id: UUID,
    job_id: int,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Unassign job (remove assignee)."""
    
    @sync_to_async
    def unassign(project, job_id):
        from jobs.models import Job
        
        try:
            job = Job.objects.get(id=job_id, project=project)
        except Job.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job.assignee = None
        job.status = 'unassigned'
        job.save(update_fields=['assignee', 'status', 'updated_at'])
        
        return job
    
    job = await unassign(project_ctx.project, job_id)
    
    return JobResponse(
        id=str(job.id),
        name=job.name,
        description=job.description,
        status=job.status,
        image_count=job.image_count,
        assignee=None,
        batch_id=job.batch_id,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat()
    )