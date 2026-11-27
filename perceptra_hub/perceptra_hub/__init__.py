# Make the configured Celery app the current one in every Django process (shell,
# management commands, Django admin), not only in the API and workers, which import
# it themselves. Otherwise a task sent from there goes to Celery's default
# (amqp://localhost) and fails with "connection refused".
from api.config.celery_utils import celery_app

__all__ = ("celery_app",)
