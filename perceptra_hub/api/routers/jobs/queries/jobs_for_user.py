from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from django.db.models import Count, Q
from users.models import CustomUser as User
from projects.models import Project
from jobs.models import Job
from api.dependencies import (
    get_current_user,
    RequestContext,
    get_request_context,
)
from api.routers.jobs.schemas import (
    JobOut, AssignedUserOut
)

router = APIRouter()

def _display_name(user) -> str:
    full = f"{user.first_name or ''} {user.last_name or ''}".strip()
    return full or user.username


@router.get("/jobs/me", response_model=List[JobOut])
def get_jobs_for_user(
    ctx:RequestContext=Depends(get_request_context),
):
    jobs = Job.objects.filter(
        assignee=ctx.user,
        project__organization=ctx.organization,
        ).exclude(
            status="completed"
            ).select_related(
                "project", "assignee", "assigned_by"  # read for every job below
            ).annotate(
                # Sent back by a reviewer, waiting for this user's fixes.
                to_fix=Count(
                    "images",
                    filter=Q(
                        images__is_active=True,
                        images__project_image__is_active=True,
                        images__project_image__status="rejected",
                    ),
                ),
            ).order_by(
                "-created_at"
            )

    return [
        JobOut(
            id=job.id,
            name=f"{job.project.name} - {job.name}",
            description=job.description,
            status=job.status,
            imageCount=job.image_count,
            assignedUser=AssignedUserOut(
                id=job.assignee.id,
                username=job.assignee.username,
                email=job.assignee.email,
                avatar=getattr(job.assignee, "avatar", None),
            ) if job.assignee else None,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            project_id=job.project.project_id,
            project_name=job.project.name,
            # A split parent keeps its image rows; its slices carry the work.
            to_fix=0 if job.status == "sliced" else job.to_fix,
            assigned_by=_display_name(job.assigned_by) if job.assigned_by else None,
            assigned_by_id=str(job.assigned_by_id) if job.assigned_by_id else None,
            assigned_at=job.assigned_at.isoformat() if job.assigned_at else None,
        )
        for job in jobs
    ]