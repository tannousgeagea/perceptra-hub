"""
Stopping a training run.

The run is marked cancelled first (so every path that reports progress sees
it and stops writing), then the compute that runs it is asked to stop:

- platform GPU: the Celery task is terminated (PlatformGPUAdapter.cancel_job);
- cloud providers: the provider's own cancel (each adapter's cancel_job);
- compute agent: the agent learns it from the reply to its next progress
  report (api/routers/agent: "cancel": true) and stops after that epoch.

A cancelled run stays cancelled: the progress and completion paths check
is_cancelled() and leave it alone.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from django.db import transaction
from django.utils import timezone

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = {"pending", "queued", "initializing", "running"}


class NotActive(Exception):
    """The run already ended (completed, failed or cancelled)."""


class TrainingStopped(Exception):
    """Raised inside a trainer's epoch callback to end a run someone stopped."""


@dataclass
class CancelOutcome:
    status: str
    # True: the compute accepted the stop; False: it couldn't be reached;
    # None: nothing was running anywhere yet (queued runs).
    compute_stopped: Optional[bool]
    # "agent" stops at the end of its current epoch; others stop now.
    stops: str
    detail: str


def is_cancelled(session_id) -> bool:
    """Fresh from the database: has someone stopped this run?"""
    from training.models import TrainingSession

    return TrainingSession.objects.filter(pk=session_id, status="cancelled").exists()


def _person(user) -> str:
    if user is None:
        return "someone"
    return f"{user.first_name or ''} {user.last_name or ''}".strip() or user.username or user.email


def _stop_compute(session) -> tuple:
    """Ask whatever runs this session to stop. Returns (stopped, stops, detail)."""
    from compute.models import TrainingJob

    job = (TrainingJob.objects.select_related("actual_provider")
           .filter(training_session=session).first())
    if job and job.external_job_id:
        from compute.adapters import get_adapter_for_provider

        is_agent = job.external_job_id.startswith("agent:")
        try:
            adapter = get_adapter_for_provider(job.actual_provider)
            ok = bool(adapter.cancel_job(job.external_job_id))
        except Exception as exc:  # provider down, bad credentials, unknown type...
            logger.warning("cancel_job failed for session %s: %s", session.pk, exc)
            return False, "now", f"Couldn't reach {job.actual_provider.name} to stop it: {exc}"
        if is_agent:
            return ok, "after_epoch", "The agent stops at the end of its current epoch."
        return ok, "now", "" if ok else f"{job.actual_provider.name} didn't confirm the stop."

    # Runs started before compute jobs existed: the Celery task id.
    if session.task_id:
        from celery.result import AsyncResult

        try:
            AsyncResult(session.task_id).revoke(terminate=True, signal="SIGTERM")
            return True, "now", ""
        except Exception as exc:
            logger.warning("revoke failed for session %s: %s", session.pk, exc)
            return False, "now", f"Couldn't stop the training task: {exc}"

    return None, "now", "It hadn't started on any compute yet."


def cancel_run(session, user) -> CancelOutcome:
    from compute.models import TrainingJob
    from training.epochs import publish_status
    from training.models import TrainingSession

    now = timezone.now()
    with transaction.atomic():
        s = (TrainingSession.objects.select_for_update()
             .select_related("model_version__model__project").get(pk=session.pk))
        if s.status not in ACTIVE_STATUSES:
            raise NotActive(s.status)
        s.status = "cancelled"
        s.completed_at = now
        s.estimated_time_remaining = None
        # Why it ended, shown on the run page (as a failure's error is).
        s.error_message = f"Stopped by {_person(user)} at epoch {s.current_epoch or 0}."
        s.save(update_fields=["status", "completed_at", "estimated_time_remaining", "error_message", "updated_at"])

        mv = s.model_version
        if mv.status in ("draft", "queued", "training"):
            mv.status = "cancelled"
            mv.error_message = s.error_message
            mv.save(update_fields=["status", "error_message", "updated_at"])
        TrainingJob.objects.filter(training_session=s, completed_at__isnull=True).update(completed_at=now)
        publish_status(s)

    # Outside the transaction: the run is already cancelled for everyone.
    stopped, stops, detail = _stop_compute(s)
    return CancelOutcome(status=s.status, compute_stopped=stopped, stops=stops, detail=detail)
