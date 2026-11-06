"""Notification housekeeping (scheduled in api/config/celery_config.py)."""
import logging
from datetime import timedelta

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)

# Read notices older than this are removed; unread ones are always kept.
READ_RETENTION_DAYS = 90


@shared_task
def prune_read_notifications():
    from notifications.models import Notification

    cutoff = timezone.now() - timedelta(days=READ_RETENTION_DAYS)
    deleted, _ = Notification.objects.filter(read_at__isnull=False, updated_at__lt=cutoff).delete()
    logger.info('Pruned %s read notifications older than %s days', deleted, READ_RETENTION_DAYS)
    return f"Deleted {deleted} read notifications"
