import time
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from django.db.models import Count, Q
from users.models import CustomUser as User
from projects.models import Project
from jobs.models import Job
from fastapi.routing import APIRoute
from fastapi import Request, Response
from typing import Callable, Optional
from api.routers.auth.queries.dependencies import (
    user_project_access_dependency,
    project_edit_admin_or_org_admin_dependency,
    project_admin_or_org_admin_dependency,
)

class TimedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            print(f"route duration: {duration}")
            print(f"route response: {response}")
            print(f"route response headers: {response.headers}")
            return response

        return custom_route_handler

router = APIRouter(
    route_class=TimedRoute
)

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

@router.get("/projects/{project_id}/jobs", response_model=List[JobOut])
def get_jobs_for_project(
    project_id: str,
    _user=Depends(project_edit_admin_or_org_admin_dependency),
):
    try:
        jobs = (
            Job.objects.filter(project__name=project_id)
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
            id=job.id,
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
