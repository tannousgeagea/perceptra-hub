import time
from fastapi import HTTPException
from fastapi import Request, Response
from fastapi.routing import APIRoute
from fastapi import APIRouter, Depends, Body
from django.shortcuts import get_object_or_404
from users.models import CustomUser as User
from organizations.models import Organization
from memberships.models import ProjectMembership
from pydantic import BaseModel
from typing import Optional, Literal
from fastapi import Path
from django.db import transaction
from jobs.models import Job, JobImage
from projects.models import ProjectImage
from api.routers.auth.queries.dependencies import (
    job_project_editor_or_admin_dependency
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


class JobStatusUpdateInput(BaseModel):
    status:Literal["in_review", "completed"]

@router.patch("/jobs/{job_id}/status")
@transaction.atomic
def update_job_status(
    job_id: int = Path(...),
    data: JobStatusUpdateInput = Body(...),
    _access=Depends(job_project_editor_or_admin_dependency),
):
    job = Job.objects.select_related("project").filter(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check completion condition if marking as completed
    if data.status == "completed":
        job_image_ids = JobImage.objects.filter(job=job).values_list("project_image_id", flat=True)
        unreviewed_count = ProjectImage.objects.filter(id__in=job_image_ids, reviewed=False).count()
        if unreviewed_count > 0:
            raise HTTPException(
                status_code=400,
                detail="Cannot mark as completed. Not all images are reviewed.",
            )

    job.status = data.status
    job.save(update_fields=["status", "updated_at"])

    return {"detail": f"Job status updated to {data.status}"}