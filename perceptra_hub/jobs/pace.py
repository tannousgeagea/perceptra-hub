"""
Team pace: how fast each member finishes images, and how much is already
waiting for them. Used to size job slices and to suggest assignees.

Pace comes from the activity log (server-side, every device), not from the
browser. Active time is the sum of gaps between a member's work events,
each gap capped at IDLE_GAP, plus a lead-in per session; so a lunch break
or an idle night does not count as slow work.
"""
from collections import defaultdict
from datetime import timedelta
from statistics import median
from typing import Dict, Iterable

from django.db.models import Count, Q
from django.utils import timezone

from activity.models import ActivityEvent, ActivityEventType
from jobs.models import Job, JobImage, JobStatus

WINDOW_DAYS = 14
IDLE_GAP = timedelta(minutes=5)          # longer gaps end a work session
SESSION_LEAD_IN = timedelta(minutes=1)   # work before a session's first event
MIN_IMAGES_FOR_PACE = 10                 # fewer finished images: pace not trusted

WORK_EVENTS = (
    ActivityEventType.ANNOTATION_CREATE,
    ActivityEventType.ANNOTATION_UPDATE,
    ActivityEventType.ANNOTATION_DELETE,
    ActivityEventType.IMAGE_REVIEW,
)

# Images still to be worked on in a job: not started, or sent back for fixes.
OPEN_STATUSES = ("unannotated", "rejected")


def _active_seconds(timestamps) -> float:
    """Active time from sorted event times: capped gaps plus a lead-in per session."""
    if not timestamps:
        return 0.0
    total = SESSION_LEAD_IN.total_seconds()
    for previous, current in zip(timestamps, timestamps[1:]):
        gap = current - previous
        total += gap.total_seconds() if gap <= IDLE_GAP else SESSION_LEAD_IN.total_seconds()
    return total


def member_pace(organization, user_ids: Iterable[int], days: int = WINDOW_DAYS) -> Dict[int, dict]:
    """
    Per user: images finished (distinct images they reviewed/approved), active
    hours, images per active hour, and active days, over the last `days`.
    """
    user_ids = list(user_ids)
    since = timezone.now() - timedelta(days=days)
    events = (
        ActivityEvent.objects.filter(
            organization=organization,
            user_id__in=user_ids,
            event_type__in=WORK_EVENTS,
            timestamp__gte=since,
        )
        .order_by("user_id", "timestamp")
        .values_list("user_id", "timestamp", "event_type", "metadata")
    )

    times = defaultdict(list)
    finished = defaultdict(set)
    for user_id, ts, event_type, metadata in events.iterator():
        times[user_id].append(ts)
        if event_type == ActivityEventType.IMAGE_REVIEW and metadata and metadata.get("project_image_id"):
            finished[user_id].add(metadata["project_image_id"])

    result = {}
    for user_id in user_ids:
        seconds = _active_seconds(times.get(user_id, []))
        images = len(finished.get(user_id, ()))
        hours = seconds / 3600
        result[user_id] = {
            "images_finished": images,
            "active_hours": round(hours, 2),
            "active_days": len({ts.date() for ts in times.get(user_id, [])}),
            "images_per_hour": round(images / hours, 1) if hours > 0 and images else None,
            "measured": images >= MIN_IMAGES_FOR_PACE,
        }
    return result


def open_workload(organization, user_ids: Iterable[int]) -> Dict[int, int]:
    """Images still to do in each user's active jobs (sliced parents excluded)."""
    rows = (
        JobImage.objects.filter(
            job__project__organization=organization,
            job__project__is_deleted=False,
            job__is_active=True,
            job__assignee_id__in=list(user_ids),
            is_active=True,
            project_image__is_active=True,
            project_image__status__in=OPEN_STATUSES,
        )
        .exclude(job__status__in=[JobStatus.SLICED, JobStatus.COMPLETED])
        .values("job__assignee_id")
        .annotate(n=Count("id"))
    )
    return {row["job__assignee_id"]: row["n"] for row in rows}


def team_pace(organization, user_ids: Iterable[int]) -> dict:
    """
    Pace and queue per user, plus the team median pace used for members
    without enough history. `hours_to_clear_queue` uses the member's own
    pace when measured, the team median otherwise.
    """
    user_ids = list(user_ids)
    pace = member_pace(organization, user_ids)
    queue = open_workload(organization, user_ids)

    measured = [p["images_per_hour"] for p in pace.values() if p["measured"] and p["images_per_hour"]]
    team_median = round(median(measured), 1) if measured else None

    members = {}
    for user_id in user_ids:
        p = pace[user_id]
        effective = p["images_per_hour"] if p["measured"] and p["images_per_hour"] else team_median
        queued = queue.get(user_id, 0)
        members[user_id] = {
            **p,
            "open_images": queued,
            "effective_images_per_hour": effective,
            "hours_to_clear_queue": round(queued / effective, 1) if effective else None,
        }
    return {"window_days": WINDOW_DAYS, "team_median_images_per_hour": team_median, "members": members}
