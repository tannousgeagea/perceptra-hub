import time
from fastapi import Request, Response
from fastapi.routing import APIRoute
from fastapi import APIRouter, Depends, Body
from django.shortcuts import get_object_or_404
from users.models import CustomUser as User
from organizations.models import Organization
from memberships.models import ProjectMembership
from pydantic import BaseModel
from typing import Optional
from fastapi import Path
from django.db import transaction
from jobs.models import Job
from api.routers.auth.queries.dependencies import (
    job_project_admin_or_org_admin_dependency
)

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
    route_class=TimedRoute
)

class JobAssignmentInput(BaseModel):
    user_id: int | None

@router.patch("/jobs/{job_id}/assign")
@transaction.atomic
def assign_user_to_job(
    job_id: int = Path(...),
    data: JobAssignmentInput = Body(...),
    project_admin=Depends(job_project_admin_or_org_admin_dependency),
):
    job = get_object_or_404(Job, id=job_id)

    if data.user_id:
        assignee = get_object_or_404(User, id=data.user_id)
        job.assignee = assignee
        if job.status == 'unassigned':
            job.status = "assigned"
    else:
        job.assignee = None
        job.status = "unassigned"

    job.save(update_fields=["assignee", "status", "updated_at"])
    return {"detail": "User assignment updated successfully."}