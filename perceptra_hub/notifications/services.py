"""
Creating notifications. Every entry point is best-effort: a failure is logged
and never breaks the write that caused it.
"""
import logging
from typing import Callable, Iterable, Optional

from django.db import transaction

from notifications.models import Notification
from realtime.bus import publish_to_user

logger = logging.getLogger(__name__)

# Job names listed in a merged "assigned you N jobs" notice.
MAX_LISTED_JOBS = 5


def display_name(user) -> str:
    if user is None:
        return 'Someone'
    full = f"{user.first_name or ''} {user.last_name or ''}".strip()
    return full or user.username


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def serialize(n: Notification) -> dict:
    return {
        'id': n.id,
        'kind': n.kind,
        'title': n.title,
        'body': n.body,
        'url': n.url,
        'count': n.count,
        'read': n.read_at is not None,
        'organization_id': str(n.organization.org_id),
        'project_id': str(n.project.project_id) if n.project_id else None,
        'actor': display_name(n.actor) if n.actor_id else None,
        'created_at': n.created_at.isoformat(),
        'updated_at': n.updated_at.isoformat(),
    }


def notify(
    *,
    recipient_id,
    organization,
    kind: str,
    build: Callable[[Optional[Notification]], dict],
    project=None,
    actor=None,
    group_key: str = '',
) -> Optional[Notification]:
    """
    Create a notice, or fold it into the recipient's unread one with the same
    `group_key` (its count goes up and it moves to the top, unread).

    `build(previous)` returns the fields to store: title, body, url, and
    optionally data. `previous` is the notice being merged into (count not yet
    incremented), or None for a new one. Nobody is notified of their own action.
    """
    if recipient_id is None or (actor is not None and actor.pk == recipient_id):
        return None
    try:
        with transaction.atomic():
            previous = None
            if group_key:
                previous = (
                    Notification.objects.select_for_update()
                    .filter(
                        recipient_id=recipient_id,
                        organization=organization,
                        group_key=group_key,
                        read_at__isnull=True,
                    )
                    .order_by('-updated_at')
                    .first()
                )
            fields = build(previous)
            if previous is not None:
                notice = previous
                notice.count += 1
                notice.actor = actor
            else:
                notice = Notification(
                    recipient_id=recipient_id,
                    organization=organization,
                    project=project,
                    actor=actor,
                    kind=kind,
                    group_key=group_key,
                )
            notice.title = fields['title'][:255]
            notice.body = fields.get('body', '')
            notice.url = fields.get('url', '')[:500]
            notice.data = fields.get('data', notice.data or {})
            notice.save()
            publish_to_user(recipient_id, 'notification', serialize(notice))
        return notice
    except Exception:
        logger.exception('Could not create %s notification for user %s', kind, recipient_id)
        return None


# ─── Events ──────────────────────────────────────────────────────────────────

def job_url(job, status: Optional[str] = None) -> str:
    url = f"/projects/{job.project.project_id}/annotate/job/{job.id}"
    return f"{url}?status={status}" if status else url


def notify_job_assigned(job, actor=None) -> None:
    """The job's assignee got it from someone else. Several in a row merge."""
    project = job.project
    giver = display_name(actor) if actor else None

    def build(previous):
        jobs = list((previous.data or {}).get('jobs', [])) if previous else []
        jobs = [j for j in jobs if j.get('id') != job.id]
        jobs.insert(0, {'id': job.id, 'name': job.name, 'images': job.image_count or 0})
        if len(jobs) == 1:
            title = f"{giver} assigned you {job.name}" if giver else f"New job assigned: {job.name}"
            body = f"{plural(job.image_count or 0, 'image')} in {project.name}."
            url = job_url(job)
        else:
            images = sum(j.get('images', 0) for j in jobs)
            names = ', '.join(j['name'] for j in jobs[:MAX_LISTED_JOBS])
            more = f" and {len(jobs) - MAX_LISTED_JOBS} more" if len(jobs) > MAX_LISTED_JOBS else ''
            title = f"{giver} assigned you {len(jobs)} jobs" if giver else f"{len(jobs)} new jobs assigned to you"
            body = f"{plural(images, 'image')} in {project.name}: {names}{more}."
            url = f"/projects/{project.project_id}/annotate"
        return {'title': title, 'body': body, 'url': url, 'data': {'jobs': jobs[:50]}}

    notify(
        recipient_id=job.assignee_id,
        organization=project.organization,
        project=project,
        actor=actor,
        kind=Notification.Kind.ASSIGNED,
        group_key=f"assigned:{project.id}:{actor.pk if actor else 0}",
        build=build,
    )


def notify_sent_back(job, reviewer, reason: str = '') -> None:
    """One image of the job went back to its assignee. Merged per job and reviewer."""
    if job is None:
        return
    who = display_name(reviewer)

    def build(previous):
        n = (previous.count + 1) if previous else 1
        note = (reason or '').strip()
        body = f"In {job.name}: “{note[:140]}”" if note else f"In {job.name}: open it to see what needs fixing."
        return {
            'title': f"{who} sent back {plural(n, 'image')}",
            'body': body,
            'url': job_url(job, 'rejected'),
        }

    notify(
        recipient_id=job.assignee_id,
        organization=job.project.organization,
        project=job.project,
        actor=reviewer,
        kind=Notification.Kind.SENT_BACK,
        group_key=f"sent_back:{job.id}:{reviewer.pk if reviewer else 0}",
        build=build,
    )


def project_reviewer_ids(project) -> set:
    """Members who can review the project; the org's admins if it names none."""
    from memberships.models import OrganizationMembership, ProjectMembership
    from memberships.roles import OrgRole, roles_for_permission

    review_roles = roles_for_permission('review', scope='project')
    ids = set(
        ProjectMembership.objects.filter(project=project, role__name__in=review_roles)
        .values_list('user_id', flat=True)
    )
    if not ids:
        ids = set(
            OrganizationMembership.objects.filter(
                organization=project.organization,
                role__name__in=[OrgRole.ADMIN.value, OrgRole.OWNER.value],
                status='active',
            ).values_list('user_id', flat=True)
        )
    return ids


def notify_ready_for_review(job, waiting: int, recipients: Iterable[int]) -> None:
    """Everything in the job is annotated (or fixed): tell the reviewers."""
    project = job.project
    finisher = job.assignee

    def build(previous):
        by = f"{display_name(finisher)} finished it" if finisher else 'All images are annotated'
        return {
            'title': f"{job.name} is ready for review",
            'body': f"{by}: {plural(waiting, 'image')} waiting in {project.name}.",
            'url': job_url(job, 'annotated'),
        }

    for user_id in recipients:
        notify(
            recipient_id=user_id,
            organization=project.organization,
            project=project,
            actor=finisher,
            kind=Notification.Kind.READY_FOR_REVIEW,
            group_key=f"ready:{job.id}",
            build=build,
        )
