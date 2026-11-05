

"""
FastAPI routes for project and job management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field
import logging
from asgiref.sync import sync_to_async
from datetime import datetime
from django.db.models import Count, Min, Q

from api.routers.projects.schemas import JobImagesResponce
from api.dependencies import get_project_context, ProjectContext
from jobs.models import Job, JobImage
from images.models import Image
from common_utils.image.utils import parse_project_image_query, apply_job_image_filters

from django.contrib.auth import get_user_model

User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects")


@sync_to_async
def get_dowload_url(image:Image, expiration:int=3600):
    return image.get_download_url(expiration=expiration)


# The job view's tabs, in display order. 'dataset' = completed (moved to a dataset).
JOB_IMAGE_TABS = ("unannotated", "rejected", "annotated", "reviewed", "dataset")


class JobImageCounts(BaseModel):
    unannotated: int = 0
    rejected: int = 0      # sent back to the annotator ("needs fixes")
    annotated: int = 0
    reviewed: int = 0
    dataset: int = 0
    marked_null: int = 0
    total: int = 0
    # All jobs only: when the longest-waiting annotated image last changed (review backlog age).
    oldest_annotated_at: Optional[datetime] = None


def count_job_images(job_images_qs) -> dict:
    """All tab counts in a single aggregate query."""
    aggregates = {
        tab: Count("id", filter=Q(project_image__status=tab)) for tab in JOB_IMAGE_TABS
    }
    aggregates["marked_null"] = Count("id", filter=Q(project_image__marked_as_null=True))
    aggregates["total"] = Count("id")
    return job_images_qs.aggregate(**aggregates)


def _get_job_or_404(project, job_id) -> Job:
    try:
        return Job.objects.get(id=job_id, project=project)
    except Job.DoesNotExist:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")


# ─── "All jobs": every image of the project's active jobs, as one pseudo-job ──
# Lets the job view and the annotator walk work across jobs (e.g. a reviewer's
# queue: All jobs → Annotated, oldest waiting first) with the same endpoints'
# shapes. Split parents are left out: their slices hold the same images.

ALL_JOBS_ID = "all"


def _all_jobs_queryset(project):
    return JobImage.objects.filter(
        job__project=project,
        job__is_active=True,
        is_active=True,
        project_image__is_active=True,
    ).exclude(job__status="sliced")


def _all_jobs_summary(project) -> dict:
    return {
        "id": ALL_JOBS_ID,
        "name": "All jobs",
        "status": "assigned",
        "assignee": None,
        "created_at": project.created_at,
        "updated_at": None,
    }


def _job_summary(job) -> dict:
    return {
        "id": str(job.id),
        "name": job.name,
        "status": job.status,
        "assignee": job.assignee.username if job.assignee else None,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _job_images_response(queryset, job_summary, *, skip, limit, status_filter, annotated, query,
                         include_counts, oldest_first=False):
    """
    One page of job images in the list response shape. `queryset` is the
    JobImage set to list (one job, or all jobs). `oldest_first` orders by when
    the image last changed, so the longest-waiting work comes first.
    """
    queryset = queryset.select_related('project_image', 'job__assignee')
    queryset = queryset.order_by('project_image__updated_at', 'id') if oldest_first else queryset.order_by('id')

    counts = count_job_images(queryset) if include_counts else {}

    if query:
        filters = parse_project_image_query(query)
        queryset = apply_job_image_filters(queryset, filters)
    else:
        if status_filter:
            queryset = queryset.filter(project_image__status=status_filter)
        if annotated is not None:
            queryset = queryset.filter(project_image__annotated=annotated)

    total = queryset.count()
    image_ids = list(queryset.values_list('project_image__image__id', flat=True))
    project_image_ids = list(queryset.values_list('project_image__id', flat=True))

    job_images = list(
        queryset.select_related(
            'project_image__image',
            'project_image__mode'
        ).prefetch_related(
            'project_image__annotations__annotation_class',
            'project_image__annotations__annotation_type'
        )[skip:skip + limit]
    )

    return {
        "total": total,
        "annotated": counts.get("annotated", 0),
        "unannotated": counts.get("unannotated", 0),
        "reviewed": counts.get("reviewed", 0),
        "rejected": counts.get("rejected", 0),
        "dataset": counts.get("dataset", 0),
        "marked_null": counts.get("marked_null", 0),
        "image_ids": list(map(str, image_ids)),
        "project_image_ids": list(map(str, project_image_ids)),
        "job": job_summary,
        "images": [
            {
                "id": str(ji.project_image.image.id),
                "project_image_id": str(ji.project_image.id),
                "image_id": str(ji.project_image.image.image_id),
                "name": ji.project_image.image.name,
                "original_filename": ji.project_image.image.original_filename,
                "width": ji.project_image.image.width,
                "height": ji.project_image.image.height,
                "aspect_ratio": ji.project_image.image.aspect_ratio,
                "file_format": ji.project_image.image.file_format,
                "file_size": ji.project_image.image.file_size,
                "file_size_mb": ji.project_image.image.file_size_mb,
                "megapixels": round(ji.project_image.image.megapixels, 2),
                "storage_key": ji.project_image.image.storage_key,
                "checksum": ji.project_image.image.checksum,
                "source_of_origin": ji.project_image.image.source_of_origin,
                "created_at": ji.project_image.image.created_at,
                "updated_at": ji.project_image.updated_at,
                "uploaded_by": ji.project_image.image.uploaded_by.username,
                "reviewed_at": ji.project_image.reviewed_at.isoformat() if ji.project_image.reviewed_at else None,
                "reviewed_by": ji.project_image.reviewed_by.username if ji.project_image.reviewed_by else None,
                "tags": [t.name for t in ji.project_image.image.tags.all()],
                "storage_profile": {
                    "id": str(ji.project_image.image.storage_profile.id),
                    "name": ji.project_image.image.storage_profile.name,
                    "backend": ji.project_image.image.storage_profile.backend
                },
                "download_url": ji.project_image.image.get_download_url(expiration=3600),
                "status": ji.project_image.status,
                "annotated": ji.project_image.annotated,
                "reviewed": ji.project_image.reviewed,
                "marked_as_null": ji.project_image.marked_as_null,
                "priority": ji.project_image.priority,
                "job_assignment_status": ji.project_image.job_assignment_status,
                "added_at": ji.project_image.added_at.isoformat(),
                "split": ji.project_image.mode.mode if ji.project_image.mode else None,
                # The job this row comes from (who annotated it), e.g. in "All jobs".
                "jobs": [_job_summary(ji.job)],
                "annotations": [
                    {
                        "id": str(ann.id),
                        "annotation_uid": ann.annotation_uid,
                        "type": ann.annotation_type.name if ann.annotation_type else None,
                        "class_id": ann.annotation_class.class_id,
                        "class_name": ann.annotation_class.name,
                        "color": ann.annotation_class.color,
                        "data": ann.data,
                        "source": ann.annotation_source,
                        "confidence": ann.confidence,
                        "reviewed": ann.reviewed,
                        "is_active": ann.is_active,
                        "created_at": ann.created_at.isoformat(),
                        "created_by": ann.created_by.username if ann.created_by else None,
                    }
                    for ann in ji.project_image.annotations.filter(is_active=True).select_related(
                        'annotation_class', 'annotation_type'
                    )
                ]
            }
            for ji in job_images
        ]
    }


@router.get(
    "/{project_id}/jobs/all/images/counts",
    summary="All Jobs: Image Counts per Tab",
    response_model=JobImageCounts,
)
async def all_jobs_image_counts(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    """Per-status counts across every active job, plus the age of the review backlog."""
    project_ctx.require_project_access()

    @sync_to_async
    def counts():
        queryset = _all_jobs_queryset(project_ctx.project)
        result = count_job_images(queryset)
        result["oldest_annotated_at"] = queryset.filter(
            project_image__status="annotated"
        ).aggregate(oldest=Min("project_image__updated_at"))["oldest"]
        return result

    return await counts()


@router.get(
    "/{project_id}/jobs/all/images",
    summary="All Jobs: List Images",
    response_model=JobImagesResponce,
)
async def list_all_jobs_images(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    include_counts: bool = Query(True),
):
    """
    Images of every active job, as one list. The work queues (unannotated,
    rejected, annotated) come oldest first, so what has waited longest is
    reviewed or fixed first.
    """
    project_ctx.require_project_access()
    oldest_first = (status or "") in ("unannotated", "rejected", "annotated") or (q or "").startswith("status:")

    @sync_to_async
    def build():
        project = project_ctx.project
        return _job_images_response(
            _all_jobs_queryset(project), _all_jobs_summary(project),
            skip=skip, limit=limit, status_filter=status, annotated=None, query=q,
            include_counts=include_counts, oldest_first=oldest_first,
        )

    return await build()


@router.get(
    "/{project_id}/jobs/{job_id}/images/counts",
    summary="Job Image Counts per Tab",
    response_model=JobImageCounts,
)
async def job_image_counts(
    project_id: UUID,
    job_id: int,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    """Per-status image counts for a job — cheap enough to poll, independent of paging."""

    @sync_to_async
    def counts():
        job = _get_job_or_404(project_ctx.project, job_id)
        return count_job_images(
            JobImage.objects.filter(job=job, project_image__is_active=True)
        )

    return await counts()


@router.get(
    "/{project_id}/jobs/{job_id}/images", 
    summary="List Job Images",
    response_model=JobImagesResponce
)
async def list_job_images(
    project_id: UUID,
    job_id: int,
    project_ctx: ProjectContext = Depends(get_project_context),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None, deprecated=True),
    annotated: Optional[bool] = Query(None, deprecated=True),
    q: Optional[str] = Query(
        None,
        description="Search query (e.g., 'status:annotated tag:car min-width:1920')",
        alias="q"
    ),
    include_counts: bool = Query(
        True,
        description="Include per-tab counts. Clients that poll /images/counts separately can pass false.",
    ),
):
    """List images assigned to a job.
    
    **Query Syntax:**
    - `status:annotated|unannotated|reviewed|dataset` - Filter by status
    - `annotated:true|false` - Filter by annotated flag
    - `reviewed:true|false` - Filter by reviewed flag
    - `marked-null:true|false` - Filter null images
    - `job-status:assigned|waiting|excluded` - Filter by job assignment
    - `tag:name` - Filter by image tag
    - `filename:text` - Filter by filename
    - `min-width:1920` - Minimum width
    - `max-width:1920` - Maximum width
    - `min-height:1080` - Minimum height
    - `max-height:1080` - Maximum height
    - `min-annotations:5` - Minimum annotation count
    - `max-annotations:10` - Maximum annotation count
    - `sort:size|name|date|width|height|annotations|priority` - Sort results
    
    **Examples:**
    - `status:annotated tag:car min-width:1920`
    - `reviewed:true sort:priority`
    - `min-annotations:5 status:annotated`
    """

    
    @sync_to_async
    def get_job_images():
        job = _get_job_or_404(project_ctx.project, job_id)
        return _job_images_response(
            JobImage.objects.filter(job=job, project_image__is_active=True),
            _job_summary(job),
            skip=skip, limit=limit, status_filter=status, annotated=annotated, query=q,
            include_counts=include_counts,
        )

    return await get_job_images()