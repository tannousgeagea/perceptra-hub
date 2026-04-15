from similarity.config.celery_utils import celery_app

celery_app.autodiscover_tasks(['similarity.tasks'])