"""
A job's status follows the state of its images.

    nothing left to do, all reviewed / in a dataset  → completed
    nothing left to annotate or fix, some to review  → in_review
    work left (unannotated, sent back), has assignee → assigned
    work left, nobody on it                          → unassigned

Sliced jobs and jobs with no images are left as they are. `sync_job_status`
is the one place that moves a job; the notifications for "ready for review"
and "completed" are sent from the transition it reports, so the status and
the notices can never disagree.
"""
import logging
from dataclasses import dataclass
from typing import Iterable, Optional

from django.db import transaction
from django.db.models import Count, Q

from jobs.models import Job, JobImage, JobStatus

logger = logging.getLogger(__name__)

PENDING_STATUSES = ('unannotated', 'in_progress', 'rejected')
WAITING_STATUSES = ('annotated',)
DONE_STATUSES = ('reviewed', 'approved', 'dataset')


@dataclass(frozen=True)
class JobProgress:
    total: int
    pending: int   # still to annotate, or sent back to fix
    waiting: int   # annotated, waiting for a reviewer
    done: int      # reviewed or in a dataset

    def as_dict(self) -> dict:
        return {'total': self.total, 'pending': self.pending, 'waiting': self.waiting, 'done': self.done}


@dataclass(frozen=True)
class Transition:
    job: Job
    previous: str
    status: str
    progress: JobProgress


def job_progress(job_id) -> JobProgress:
    counts = JobImage.objects.filter(job_id=job_id, is_active=True, project_image__is_active=True).aggregate(
        total=Count('id'),
        pending=Count('id', filter=Q(project_image__status__in=PENDING_STATUSES)),
        waiting=Count('id', filter=Q(project_image__status__in=WAITING_STATUSES)),
        done=Count('id', filter=Q(project_image__status__in=DONE_STATUSES)),
    )
    return JobProgress(**{k: v or 0 for k, v in counts.items()})


def derive_status(job: Job, progress: JobProgress) -> str:
    """The status the job should have; its current one when nothing decides."""
    if job.status == JobStatus.SLICED or progress.total == 0:
        return job.status
    if progress.pending == 0 and progress.waiting == 0:
        return JobStatus.COMPLETED
    if progress.pending == 0:
        return JobStatus.IN_REVIEW
    return JobStatus.ASSIGNED if job.assignee_id else JobStatus.UNASSIGNED


def sync_job_status(job_id) -> Optional[Transition]:
    """
    Bring one job's status in line with its images. Returns the transition,
    or None when the status was already right. The row is locked so two
    reviewers finishing at once move the job (and notify) only once.
    """
    with transaction.atomic():
        job = (
            Job.objects.select_for_update(of=('self',))
            .select_related('project__organization', 'assignee', 'assigned_by')
            .filter(pk=job_id, is_active=True)
            .first()
        )
        if job is None or job.status == JobStatus.SLICED:
            return None
        progress = job_progress(job.pk)
        status = derive_status(job, progress)
        if status == job.status:
            return None
        previous = job.status
        job.status = status
        job.save(update_fields=['status', 'updated_at'])
    return Transition(job=job, previous=previous, status=status, progress=progress)


def jobs_of_image(project_image_id) -> list:
    return list(
        JobImage.objects.filter(project_image_id=project_image_id, is_active=True)
        .exclude(job__status=JobStatus.SLICED)
        .values_list('job_id', flat=True)
        .distinct()
    )


def sync_jobs(job_ids: Iterable, *, actor_id=None, announce: bool = True) -> list:
    """
    Sync several jobs; for each that moved, push the change to the project
    and (unless `announce` is off, as in a backfill) send its notices.
    Best-effort: a failure is logged per job and never raised.
    """
    transitions = []
    for job_id in job_ids:
        try:
            transition = sync_job_status(job_id)
        except Exception:
            logger.exception('status sync for job %s failed', job_id)
            continue
        if transition is None:
            continue
        transitions.append(transition)
        _publish(transition)
        if announce:
            _announce(transition, actor_id)
    return transitions


def _publish(t: Transition) -> None:
    from realtime.bus import publish

    publish(t.job.project.project_id, 'job_status', {
        'job_id': t.job.pk,
        'name': t.job.name,
        'status': t.status,
        'previous': t.previous,
        'progress': t.progress.as_dict(),
    })


def _announce(t: Transition, actor_id) -> None:
    from django.contrib.auth import get_user_model

    from notifications.services import notify_job_completed, notify_ready_for_review, project_reviewer_ids

    try:
        if t.status == JobStatus.IN_REVIEW:
            recipients = project_reviewer_ids(t.job.project) - {t.job.assignee_id}
            notify_ready_for_review(t.job, t.progress.waiting, recipients)
        elif t.status == JobStatus.COMPLETED:
            actor = get_user_model().objects.filter(pk=actor_id).first() if actor_id else None
            notify_job_completed(t.job, t.progress.done, actor=actor)
    except Exception:
        logger.exception('notices for job %s (%s → %s) failed', t.job.pk, t.previous, t.status)
