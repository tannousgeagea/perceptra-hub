import time
from uuid import UUID
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
from api.dependencies import (
    get_current_user,
    RequestContext,
    get_request_context,
)
from api.routers.jobs.schemas import (
    JobOut, AssignedUserOut
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

@router.get("/jobs/me", response_model=List[JobOut])
def get_jobs_for_user(
    ctx:RequestContext=Depends(get_request_context),
):
    jobs = Job.objects.filter(
        assignee=ctx.user,
        project__organization=ctx.organization,
        ).exclude(
            status="completed"
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
        )
        for job in jobs
    ]