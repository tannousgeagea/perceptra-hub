"""
Notifications from model changes, so every code path is covered (single and
bulk review, job create/assign/slice, admin). Work runs after the commit and
never raises into the write.

- Job gets a new assignee        → "X assigned you Job N"      (to the assignee)
- ImageReview, not approved      → "X sent back N images"      (to the job's assignee)
- Image becomes 'annotated' and
  its job has nothing left to do → "Job N is ready for review" (to the reviewers)
"""
import logging

from django.db import transaction
from django.db.models import Count, Q
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver

from jobs.models import Job, JobImage
from projects.models import ImageReview, ProjectImage

logger = logging.getLogger(__name__)

PENDING_STATUSES = ('unannotated', 'rejected')


def _touches(update_fields, field: str) -> bool:
    return update_fields is None or field in update_fields


# ─── Job assigned ────────────────────────────────────────────────────────────

@receiver(pre_save, sender=Job, dispatch_uid='notifications_job_pre_save')
def remember_assignee(sender, instance, raw=False, update_fields=None, **kwargs):
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
    if raw or instance.pk is None or not _touches(update_fields, 'status'):
        return
    instance._previous_status = (
        ProjectImage.objects.filter(pk=instance.pk).values_list('status', flat=True).first()
    )


@receiver(post_save, sender=ProjectImage, dispatch_uid='notifications_image_saved')
def image_saved(sender, instance, created, raw=False, **kwargs):
    if raw or created or instance.status != 'annotated':
        return
    if getattr(instance, '_previous_status', 'annotated') == 'annotated':
        return  # not a change into 'annotated'
    image_id = instance.pk
    transaction.on_commit(lambda: _check_jobs_ready(image_id))


def _check_jobs_ready(project_image_id) -> None:
    from notifications.services import notify_ready_for_review, project_reviewer_ids

    try:
        job_ids = list(
            JobImage.objects.filter(project_image_id=project_image_id, is_active=True)
            .exclude(job__status='sliced')
            .values_list('job_id', flat=True)
        )
        for job in Job.objects.filter(pk__in=job_ids).select_related('project__organization', 'assignee'):
            counts = JobImage.objects.filter(job=job, is_active=True, project_image__is_active=True).aggregate(
                pending=Count('id', filter=Q(project_image__status__in=PENDING_STATUSES)),
                waiting=Count('id', filter=Q(project_image__status='annotated')),
            )
            if counts['pending'] or not counts['waiting']:
                continue
            recipients = project_reviewer_ids(job.project) - {job.assignee_id}
            notify_ready_for_review(job, counts['waiting'], recipients)
    except Exception:
        logger.exception('ready-for-review check for image %s failed', project_image_id)
