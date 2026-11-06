"""
Notifications from model changes, so every code path is covered (single and
bulk review, job create/assign/slice, admin). Work runs after the commit and
never raises into the write.

- Job gets a new assignee        → "X assigned you Job N"      (to the assignee)
- ImageReview, not approved      → "X sent back N images"      (to the job's assignee)

Job status follows its images (jobs.lifecycle). An image status change, a
job image added or removed, or a new assignee re-syncs the job; the
transition sends its own notice:
- job → in_review                → "Job N is ready for review" (to the reviewers)
- job → completed                → "Job N is complete"         (to the assignee and assigner)
"""
import logging

from django.db import transaction
from django.db.models.signals import post_delete, post_save, pre_save
from django.dispatch import receiver

from jobs.lifecycle import jobs_of_image, sync_jobs
from jobs.models import Job, JobImage
from projects.models import ImageReview, ProjectImage

logger = logging.getLogger(__name__)


def _touches(update_fields, field: str) -> bool:
    return update_fields is None or field in update_fields


# ─── Job assigned ────────────────────────────────────────────────────────────

@receiver(pre_save, sender=Job, dispatch_uid='notifications_job_pre_save')
def remember_assignee(sender, instance, raw=False, update_fields=None, **kwargs):
    # Forget the previous save's value: only this save's change counts.
    instance.__dict__.pop('_previous_assignee_id', None)
    if raw or instance.pk is None or not _touches(update_fields, 'assignee'):
        return
    instance._previous_assignee_id = (
        Job.objects.filter(pk=instance.pk).values_list('assignee_id', flat=True).first()
    )


@receiver(post_save, sender=Job, dispatch_uid='notifications_job_saved')
def job_saved(sender, instance, created, raw=False, update_fields=None, **kwargs):
    if raw or not instance.assignee_id or instance.status == 'sliced':
        return
    if not created and not hasattr(instance, '_previous_assignee_id'):
        return  # assignee not part of this save
    if not created and instance._previous_assignee_id == instance.assignee_id:
        return
    # Who gave it: the assigner, else whoever created/updated the job (an
    # uploader's own batch job is theirs: no notice to themselves).
    actor_id = instance.assigned_by_id or (instance.created_by_id if created else instance.updated_by_id)
    job_id = instance.pk

    def send():
        from notifications.services import notify_job_assigned
        from django.contrib.auth import get_user_model

        try:
            job = Job.objects.select_related('project__organization').get(pk=job_id)
            if job.assignee_id is None or job.status == 'sliced':
                return
            actor = get_user_model().objects.filter(pk=actor_id).first() if actor_id else None
            notify_job_assigned(job, actor=actor)
        except Exception:
            logger.exception('assigned notification for job %s failed', job_id)

    transaction.on_commit(send)


@receiver(post_save, sender=Job, dispatch_uid='jobs_status_on_assignee')
def job_assignee_changed(sender, instance, created, raw=False, **kwargs):
    """A job given to someone, or taken back: assigned ↔ unassigned (when work is left)."""
    if raw or created or not hasattr(instance, '_previous_assignee_id'):
        return
    if instance._previous_assignee_id == instance.assignee_id:
        return
    job_id = instance.pk
    transaction.on_commit(lambda: sync_jobs([job_id]))


# ─── Work sent back ──────────────────────────────────────────────────────────

@receiver(post_save, sender=ImageReview, dispatch_uid='notifications_review_saved')
def review_saved(sender, instance, created, raw=False, **kwargs):
    if raw or not created or instance.approved or not instance.job_id:
        return
    review_id = instance.pk

    def send():
        from notifications.services import notify_sent_back

        try:
            review = ImageReview.objects.select_related('job__project__organization', 'reviewer').get(pk=review_id)
            notify_sent_back(review.job, review.reviewer, review.reason)
        except Exception:
            logger.exception('sent-back notification for review %s failed', review_id)

    transaction.on_commit(send)


# ─── Ready for review ────────────────────────────────────────────────────────

@receiver(pre_save, sender=ProjectImage, dispatch_uid='notifications_image_pre_save')
def remember_status(sender, instance, raw=False, update_fields=None, **kwargs):
    instance.__dict__.pop('_previous_status', None)
    if raw or instance.pk is None or not _touches(update_fields, 'status'):
        return
    instance._previous_status = (
        ProjectImage.objects.filter(pk=instance.pk).values_list('status', flat=True).first()
    )


@receiver(post_save, sender=ProjectImage, dispatch_uid='jobs_status_on_image_saved')
def image_saved(sender, instance, created, raw=False, **kwargs):
    """The image changed status: its jobs may now be ready, complete, or back to work."""
    if raw or created or not hasattr(instance, '_previous_status'):
        return
    if instance._previous_status == instance.status:
        return
    image_id = instance.pk
    # Whoever reviewed it closes the job, if this was the last one.
    actor_id = instance.reviewed_by_id if instance.status in ('reviewed', 'approved', 'dataset') else None
    transaction.on_commit(lambda: _sync_image_jobs(image_id, actor_id))


def _sync_image_jobs(project_image_id, actor_id=None) -> None:
    try:
        sync_jobs(jobs_of_image(project_image_id), actor_id=actor_id)
    except Exception:
        logger.exception('job status sync for image %s failed', project_image_id)


# ─── Images added to / removed from a job ────────────────────────────────────

@receiver(post_save, sender=JobImage, dispatch_uid='jobs_status_on_job_image_saved')
def job_image_saved(sender, instance, created, raw=False, update_fields=None, **kwargs):
    if raw or not (created or _touches(update_fields, 'is_active')):
        return
    job_id = instance.job_id
    transaction.on_commit(lambda: sync_jobs([job_id]))


@receiver(post_delete, sender=JobImage, dispatch_uid='jobs_status_on_job_image_deleted')
def job_image_deleted(sender, instance, **kwargs):
    job_id = instance.job_id
    transaction.on_commit(lambda: sync_jobs([job_id]))
