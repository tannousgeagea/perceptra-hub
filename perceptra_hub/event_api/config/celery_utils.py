from celery import current_app as c_app
from .celery_config import settings, BaseConfig
from celery.result import AsyncResult


def create_celery():
    celery_app = c_app
    celery_app.config_from_object(settings, namespace='CELERY')
    celery_app.conf.update(task_track_started=True)
    celery_app.conf.update(task_serializer=BaseConfig.TASK_SERIALIZE)
    celery_app.conf.update(result_serializer=BaseConfig.RESULT_SERIALIZE)
    celery_app.conf.update(accept_content=BaseConfig.ACCEPT_CONTENT)
    celery_app.conf.update(result_expires=200)
    celery_app.conf.update(result_persistent=False)
    celery_app.conf.update(worker_send_task_events=False)
    celery_app.conf.update(worker_prefetch_multiplier=1)

    return celery_app


def get_task_info(task_id):
    """
    Retrieve information about a Celery task given its task ID.
    """
    
    task = AsyncResult(task_id)
    return {
        'task_id': task.id,
        'status': task.status,
        'result': task.result
    }
