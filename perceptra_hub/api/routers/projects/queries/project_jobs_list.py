
"""
FastAPI routes for project and job management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
import logging
from uuid import UUID
from asgiref.sync import sync_to_async
from django.db.models import Count, Q
from api.dependencies import get_project_context, ProjectContext
from projects.models import (
    Project
)
from django.contrib.auth import get_user_model
from jobs.models import Job, JobImage
from api.routers.projects.schemas import JobResponse, AssignedUserOut, JobProgress

User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Projects"])



@router.get("/{project_id}/jobs", summary="List Jobs")
async def list_jobs(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
    status: Optional[str] = Query(None)
):
    """List jobs for project."""
    
    @sync_to_async
    def get_jobs(project, status_filter):
        
        queryset = (
            Job.objects.filter(
                project=project,
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
        
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        return list(queryset.select_related('assignee'))
    
    jobs = await get_jobs(project_ctx.project, status)
    
    return {
        "jobs": [
            JobResponse(
                id=str(j.id),
                name=j.name,
                description=j.description,
                status=j.status,
                image_count=j.image_count,
                assignee={"id": str(j.assignee.id), "username": j.assignee.username} if j.assignee else None,
                assignedUser=AssignedUserOut(
                    id=j.assignee.id,
                    username=j.assignee.username,
                    email=j.assignee.email,
                    avatar=getattr(j.assignee, "avatar", None),
                ) if j.assignee else None,
                batch_id=j.batch_id,
                created_at=j.created_at.isoformat(),
                updated_at=j.updated_at.isoformat(),
                progress=JobProgress(
                    total=j.total,
                    annotated=j.annotated,
                    reviewed=j.reviewed,
                    completed=j.completed,
                )
            )
            for j in jobs
        ]
    }