"""
Where an image sits inside a job — so a deep link (e.g. "open in annotator"
from the Dataset page) lands the annotator on the right page of the job, with
working previous / next.

    GET /api/v1/projects/{project_id}/jobs/{job_id}/images/{image_id}/position?limit=20

The position is computed in the same list the job view shows: the job's
images ordered by id, filtered by the image's own status tab
(unannotated / annotated / reviewed; any other status → the unfiltered list).
"""
from typing import Optional
from uuid import UUID

from asgiref.sync import sync_to_async
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from api.dependencies import ProjectContext, get_project_context
from jobs.models import Job, JobImage

router = APIRouter(prefix="/projects")

STATUS_TABS = ("unannotated", "annotated", "reviewed")


class JobImagePosition(BaseModel):
    job_id: str
    job_name: str
    image_id: str
    project_image_id: str
    status: Optional[str]  # the job-view tab the image is listed under (None = all)
    index: int             # 1-based, within that tab
    page: int              # 1-based, for `limit` images per page
    limit: int
    total: int             # images in that tab


@router.get(
    "/{project_id}/jobs/{job_id}/images/{image_id}/position",
    response_model=JobImagePosition,
    summary="Position of an image within a job (for deep links)",
)
async def job_image_position(
    project_id: UUID,
    job_id: int,
    image_id: str,
    limit: int = Query(20, ge=1, le=1000, description="Page size of the job view"),
    project_ctx: ProjectContext = Depends(get_project_context),
):
    @sync_to_async
    def locate():
        job = Job.objects.filter(id=job_id, project=project_ctx.project).first()
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        member = (
            JobImage.objects.filter(job=job, project_image__image__image_id=image_id, project_image__is_active=True)
            .select_related("project_image").first()
        )
        if member is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image is not in this job")

        tab = member.project_image.status if member.project_image.status in STATUS_TABS else None
        # Same base list and order as GET /jobs/{job_id}/images.
        ordered = JobImage.objects.filter(job=job, project_image__is_active=True).order_by("id")
        if tab:
            ordered = ordered.filter(project_image__status=tab)
        ids = list(ordered.values_list("id", flat=True))
        index = ids.index(member.id) + 1
        return JobImagePosition(
            job_id=str(job.id),
            job_name=job.name,
            image_id=image_id,
            project_image_id=str(member.project_image_id),
            status=tab,
            index=index,
            page=(index - 1) // limit + 1,
            limit=limit,
            total=len(ids),
        )

    return await locate()
