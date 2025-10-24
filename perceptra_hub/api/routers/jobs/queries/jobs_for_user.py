import time
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from django.db.models import Count
from users.models import CustomUser as User
from projects.models import Project
from jobs.models import Job
from fastapi.routing import APIRoute
from fastapi import Request, Response
from typing import Callable, Optional
from api.routers.auth.queries.dependencies import (
    user_project_access_dependency,
    project_admin_or_org_admin_dependency,
    get_current_user
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

class JobOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    status: str
    imageCount: int
    assignedUser: Optional[AssignedUserOut] = None
    createdAt: str
    updatedAt: str
    projectId: str

@router.get("/jobs/my-assigned", response_model=List[JobOut])
def get_jobs_for_user(
    user=Depends(get_current_user),
):
    jobs = Job.objects.filter(assignee=user).exclude(status="completed").order_by("-created_at")

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
            createdAt=job.created_at.isoformat(),
            updatedAt=job.updated_at.isoformat(),
            projectId=job.project.name,
        )
        for job in jobs
    ]