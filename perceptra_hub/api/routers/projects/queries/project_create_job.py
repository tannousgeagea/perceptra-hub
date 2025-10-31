
"""
FastAPI routes for project and job management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
import logging
from uuid import UUID
from asgiref.sync import sync_to_async

from api.dependencies import get_project_context, ProjectContext
from projects.models import (
    Project
)
from django.contrib.auth import get_user_model
from jobs.models import Job, JobImage
from api.routers.projects.schemas import JobResponse, JobCreate

User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Projects"])

@router.post(
    "/{project_id}/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Job"
)
async def create_job(
    project_id: UUID,
    data: JobCreate,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    project_ctx.require_project_role("owner", "admin")
    """Create a new job for project."""
    @sync_to_async
    def create_job_record(project, data):

        # Get assignee if specified
        assignee = None
        if data.assignee_id:
            try:
                assignee = User.objects.get(id=data.assignee_id)
            except User.DoesNotExist:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Assignee not found"
                )
        
        # Create job
        job = Job.objects.create(
            project=project,
            name=data.name,
            description=data.description,
            assignee=assignee,
            batch_id=data.batch_id,
            status='assigned' if assignee else 'unassigned'
        )
        
        return job
    
    job = await create_job_record(project_ctx.project, data)
    
    return JobResponse(
        id=str(job.id),
        name=job.name,
        description=job.description,
        status=job.status,
        image_count=job.image_count,
        assignee={"id": str(job.assignee.id), "username": job.assignee.username} if job.assignee else None,
        batch_id=job.batch_id,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat()
    )