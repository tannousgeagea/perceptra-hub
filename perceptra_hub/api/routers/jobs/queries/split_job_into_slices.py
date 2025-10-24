from fastapi import APIRouter, Depends, HTTPException, Body, Path
from django.db import transaction
from django.shortcuts import get_object_or_404
from typing import List, Optional
from pydantic import BaseModel

from jobs.models import Job, JobImage
from users.models import CustomUser as User
from api.routers.auth.queries.dependencies import job_project_editor_or_admin_dependency

router = APIRouter()

class JobSliceInput(BaseModel):
    number_of_slices: int
    user_assignments: List[Optional[int]]  # List of user IDs or null

@router.post("/jobs/{job_id}/split")
@transaction.atomic
def split_job_into_slices(
    job_id: int = Path(...),
    payload: JobSliceInput = Body(...),
    _access=Depends(job_project_editor_or_admin_dependency)
):
    job = get_object_or_404(Job.objects.select_related("project"), id=job_id)
    job_images = list(JobImage.objects.filter(job=job))
    total_images = len(job_images)

    if payload.number_of_slices > total_images:
        raise HTTPException(status_code=400, detail="Cannot split into more slices than images")

    images_per_slice = total_images // payload.number_of_slices
    remainder = total_images % payload.number_of_slices

    slice_start = 0
    for i in range(payload.number_of_slices):
        slice_image_count = images_per_slice + (1 if i < remainder else 0)
        slice_images = job_images[slice_start:slice_start + slice_image_count]
        slice_start += slice_image_count

        assignee_id = payload.user_assignments[i]
        assignee = User.objects.filter(id=assignee_id).first() if assignee_id else None

        job_slice = Job.objects.create(
            project=job.project,
            name=f"{job.name} (Slice {i + 1})",
            description=job.description,
            status="assigned" if assignee else "unassigned",
            assignee=assignee,
            image_count=slice_image_count,
        )

        # Assign images
        JobImage.objects.bulk_create([
            JobImage(job=job_slice, project_image=ji.project_image)
            for ji in slice_images
        ])

    # Optionally: delete the original job or mark it as sliced
    job.status = "sliced"
    job.save(update_fields=["status"])

    return {"detail": f"Job split into {payload.number_of_slices} slices successfully"}
