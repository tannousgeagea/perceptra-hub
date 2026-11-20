"""
A job made of chosen images (e.g. the ones active learning ranks highest).

    POST /api/v1/projects/{project_id}/jobs/from-images

Project managers only, like creating, assigning and splitting jobs. Images
already in an open job are left where they are and reported, so nobody ends
up with the same image in two jobs.
"""
from typing import List, Optional
from uuid import UUID

from asgiref.sync import sync_to_async
from django.db import transaction
from django.utils import timezone
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import ProjectContext, get_project_context

router = APIRouter(prefix="/projects", tags=["Projects"])

MAX_IMAGES = 500
OPEN_JOB_STATUSES = ("unassigned", "assigned", "in_review")


class JobFromImages(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    assignee_id: Optional[int] = None
    project_image_ids: List[int] = Field(..., min_length=1, max_length=MAX_IMAGES)


class JobFromImagesOut(BaseModel):
    job_id: int
    name: str
    status: str
    image_count: int
    # Chosen images already in an open job (left there).
    skipped_in_open_job: List[int]
    # Ids that aren't active images of this project.
    not_found: List[int]


@router.post(
    "/{project_id}/jobs/from-images",
    response_model=JobFromImagesOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a job from chosen images",
)
async def create_job_from_images(
    project_id: UUID,
    data: JobFromImages,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    project_ctx.require_manage_permission()

    @sync_to_async
    def _create():
        from django.contrib.auth import get_user_model
        from jobs.lifecycle import sync_jobs
        from jobs.models import Job, JobImage
        from memberships.models import OrganizationMembership
        from projects.models import ProjectImage

        project = project_ctx.project
        user = project_ctx.user
        wanted = list(dict.fromkeys(data.project_image_ids))

        assignee = None
        if data.assignee_id is not None:
            is_member = OrganizationMembership.objects.filter(
                organization=project.organization, user_id=data.assignee_id, status="active"
            ).exists()
            if not is_member:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                    detail="The assignee isn't an active member of this organization")
            assignee = get_user_model().objects.get(id=data.assignee_id)

        with transaction.atomic():
            found = set(
                ProjectImage.objects.filter(project=project, is_active=True, id__in=wanted)
                .values_list("id", flat=True)
            )
            busy = set(
                JobImage.objects.filter(
                    project_image_id__in=found, is_active=True,
                    job__is_active=True, job__status__in=OPEN_JOB_STATUSES,
                ).values_list("project_image_id", flat=True)
            )
            free = [pid for pid in wanted if pid in found and pid not in busy]
            if not free:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="None of the chosen images can be added: they're already in open jobs or not in this project",
                )

            job = Job.objects.create(
                project=project,
                name=data.name.strip(),
                description=(data.description or "").strip() or None,
                assignee=assignee,
                status="assigned" if assignee else "unassigned",
                image_count=len(free),
                created_by=user,
                updated_by=user,
                assigned_by=user if assignee else None,
                assigned_at=timezone.now() if assignee else None,
            )
            JobImage.objects.bulk_create([
                JobImage(job=job, project_image_id=pid, created_by=user) for pid in free
            ])
            ProjectImage.objects.filter(id__in=free).update(job_assignment_status="assigned")
            # Images may already be annotated (bulk_create sends no signals): let the
            # job's status catch up, and tell the assignee about their new job.
            transaction.on_commit(lambda: sync_jobs([job.id], actor_id=user.id))

        return JobFromImagesOut(
            job_id=job.id,
            name=job.name,
            status=job.status,
            image_count=len(free),
            skipped_in_open_job=sorted(busy),
            not_found=[pid for pid in wanted if pid not in found],
        )

    return await _create()
