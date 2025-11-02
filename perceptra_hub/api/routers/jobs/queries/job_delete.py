"""
FastAPI route for deleting a job.
"""

import time
from fastapi.routing import APIRoute
from fastapi import Request, Response, APIRouter, Depends, HTTPException, status
from uuid import UUID
from asgiref.sync import sync_to_async
import logging

from api.dependencies import get_project_context, ProjectContext
from jobs.models import Job, JobImage
from projects.models import ProjectImage

logger = logging.getLogger(__name__)

class TimedRoute(APIRoute):
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request):
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            return response

        return custom_route_handler


router = APIRouter(
    prefix="/projects",
    route_class=TimedRoute
)


@router.delete(
    "/{project_id}/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Job",
    description="Deletes a job and unassigns all associated images. Supports soft or hard delete."
)
async def delete_job(
    project_id: UUID,
    job_id: int,
    hard_delete: bool = False,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Delete a job from a project (soft delete by default)."""

    @sync_to_async
    def delete_job_record(project, job_id, hard_delete: bool, user=None):
        from django.db import transaction

        try:
            job = Job.objects.select_related("project").get(
                id=job_id,
                project=project
            )
        except Job.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found in this project"
            )

        # Get job images
        job_images = JobImage.objects.filter(job=job).select_related("project_image")

        with transaction.atomic():
            if hard_delete:
                # Hard delete job and its related job-image links
                job.delete()
                logger.info(f"Hard deleted Job {job_id} and its job-image links")
            else:
                # Soft delete: mark related images as unassigned
                project_image_ids = [ji.project_image.id for ji in job_images]
                ProjectImage.objects.filter(id__in=project_image_ids).update(
                    job_assignment_status="unassigned"
                )

                # Then delete job (safe to cascade or keep audit)
                job.soft_delete(user=user)

        return True

    await delete_job_record(project_ctx.project, job_id, hard_delete, user=project_ctx.user)
