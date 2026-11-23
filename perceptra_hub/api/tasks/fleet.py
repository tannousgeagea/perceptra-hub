"""
Fleet background work, on the existing `activity` worker (names prefixed
"activity:" route there, api/config/celery_config.py).
"""
import logging
import os
from datetime import timedelta

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)

HEALTH_RETENTION_DAYS = int(os.getenv("FLEET_HEALTH_RETENTION_DAYS", "7"))


@shared_task(name="activity:fleet_prepare_rollout", bind=True, max_retries=3, default_retry_delay=30)
def prepare_rollout(self, rollout_pk: int):
    """Checksum the version's files, then start the rollout (or mark it failed)."""
    from fleet import artifacts, services
    from fleet.models import Rollout

    rollout = Rollout.objects.select_related("version__storage_profile", "version__model").filter(pk=rollout_pk).first()
    if rollout is None or rollout.status != Rollout.Status.PREPARING:
        return
    artifacts.prepare_version(rollout.version)
    if not artifacts.is_prepared(rollout.version) and self.request.retries < self.max_retries:
        # Storage hiccups are common at the edge of a network; try a few times.
        raise self.retry()
    services.start(rollout)


@shared_task(name="activity:fleet_advance_rollouts")
def advance_rollouts():
    """Every minute: canary timers expire even when no device calls."""
    from fleet import services
    return services.advance_all()


@shared_task(name="activity:fleet_prune_health")
def prune_health_samples():
    from fleet.models import HealthSample
    cutoff = timezone.now() - timedelta(days=HEALTH_RETENTION_DAYS)
    deleted, _ = HealthSample.objects.filter(at__lt=cutoff).delete()
    logger.info("fleet: pruned %s health samples older than %s days", deleted, HEALTH_RETENTION_DAYS)
    return deleted
