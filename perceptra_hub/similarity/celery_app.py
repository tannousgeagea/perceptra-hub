# Similarity tasks run on the main API Celery app.
# This module is kept for backwards compatibility — do not instantiate a
# separate Celery app here; that would create an isolated broker/queue that
# the main workers cannot consume from.
#
# Task discovery is handled in api/main.py:
#   celery.autodiscover_tasks(['api.tasks', 'similarity.tasks'])
