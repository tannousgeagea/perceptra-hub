"""Celery Beat task: evaluate all active retraining policies every hour."""

import logging

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(queue='activity', name='api.tasks.retraining.evaluate_retraining_policies')
def evaluate_retraining_policies():
    """
    Hourly task: check all active RetrainingPolicy records and trigger
    training runs for any that meet their threshold conditions.
    """
    from training.retraining_service import RetrainingService

    service = RetrainingService()
    triggered = service.evaluate_all_policies()
    logger.info("evaluate_retraining_policies: %d policies triggered", len(triggered))
    return {"triggered_count": len(triggered), "policies": triggered}
