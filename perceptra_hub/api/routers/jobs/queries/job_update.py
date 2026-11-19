# api/jobs.py


"""
FastAPI routes for project and job management.
"""
from fastapi import Body
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

router = APIRouter(
    prefix="/projects",
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
    # Renaming or reassigning a job is a project admin's call. The status alone
    # may also be moved by a reviewer, or by the job's assignee (an annotator).
    changes_details = any(v is not None for v in (data.name, data.description, data.assignee_id))
    if changes_details:
        project_ctx.require_manage_permission()
    only_if_assignee = not changes_details and not project_ctx.can_review()
    if only_if_assignee:
        project_ctx.require_annotate_permission()

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

        if only_if_assignee and job.assignee_id != getattr(user, "id", None):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the job's assignee or a reviewer can change its status",
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