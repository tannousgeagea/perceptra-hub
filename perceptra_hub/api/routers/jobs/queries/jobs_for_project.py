from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from django.db.models import Count, Q
from users.models import CustomUser as User
from projects.models import Project
from jobs.models import Job
from api.dependencies import (
    RequestContext,
    get_request_context
)

router = APIRouter()

class AssignedUserOut(BaseModel):
    id: int
    username: str
    email: str
    avatar: Optional[str] = None

class JobProgressOut(BaseModel):
    total: int
    annotated: int
    reviewed: int
    completed: int

class JobOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    status: str
    imageCount: int
    assignedUser: Optional[AssignedUserOut] = None
    createdAt: str
    updatedAt: str
    progress: JobProgressOut   # 👈 new field

@router.get("/jobs/project/{project_id}", response_model=List[JobOut])
def get_jobs_for_project(
    project_id: str,
    ctx:RequestContext=Depends(get_request_context),
):
    try:
        jobs = (
            Job.objects.filter(
                project__project_id=project_id,
                project__organization=ctx.organization
            )
                .select_related("assignee")
                .annotate(
                    total=Count("images"),
                    annotated=Count("images", filter=Q(images__project_image__status="annotated")),
                    reviewed=Count("images", filter=Q(images__project_image__status="reviewed")),
                    completed=Count("images", filter=Q(images__project_image__status="dataset")),
                )
                .order_by('-created_at')
            )
    except Project.DoesNotExist:
        raise HTTPException(status_code=404, detail="Project not found")

    return [
        JobOut(
            id=str(job.id),
            name=job.name,
            description=job.description,
            status=job.status,
            imageCount=job.image_count,
            assignedUser=AssignedUserOut(
                id=job.assignee.id,
                username=job.assignee.username,
                email=job.assignee.email,
                avatar=getattr(job.assignee, "avatar", None),
            ) if job.assignee else None,
            createdAt=job.created_at.isoformat(),
            updatedAt=job.updated_at.isoformat(),
            progress=JobProgressOut(
                total=job.total,
                annotated=job.annotated,
                reviewed=job.reviewed,
                completed=job.completed,
            )
        )
        for job in jobs
    ]
