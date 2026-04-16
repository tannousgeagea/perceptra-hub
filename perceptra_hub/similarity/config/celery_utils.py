# celery_utils.py

from celery import Celery
from celery.result import AsyncResult
from .celery_config import settings
from typing import Dict, Optional, List


def create_celery(app_name: str = 'worker') -> Celery:
    """
    Create and configure Celery application.
    
    Args:
        app_name: Name of the Celery application
    
    Returns:
        Configured Celery instance
    """
    celery_app = Celery(app_name)
    
    # Load config from settings object
    celery_app.config_from_object(settings, namespace='CELERY')
    
    return celery_app


def get_task_info(task_id: str) -> Dict:
    """
    Get detailed information about a Celery task.
    
    Args:
        task_id: Celery task ID
    
    Returns:
        Dictionary with task status and result
    """
    task = AsyncResult(task_id)
    
    info = {
        'task_id': task.id,
        'status': task.status,
        'result': task.result if task.ready() else None,
    }
    
    # Add additional info based on status
    if task.failed():
        info['error'] = str(task.result)
        info['traceback'] = task.traceback
    elif task.successful():
        info['completed_at'] = task.date_done
    
    return info


def get_task_status(task_id: str) -> str:
    """
    Get task status only.
    
    Args:
        task_id: Celery task ID
    
    Returns:
        Task status string (PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED)
    """
    return AsyncResult(task_id).status


def revoke_task(task_id: str, terminate: bool = False) -> Dict:
    """
    Revoke (cancel) a task.
    
    Args:
        task_id: Celery task ID
        terminate: If True, terminate the task if it's running
    
    Returns:
        Dictionary with revocation status
    """
    task = AsyncResult(task_id)
    task.revoke(terminate=terminate)
    
    return {
        'task_id': task_id,
        'revoked': True,
        'terminated': terminate
    }


def get_active_tasks(queue: Optional[str] = None) -> List[Dict]:
    """
    Get list of currently active tasks.
    
    Args:
        queue: Optional queue name to filter by
    
    Returns:
        List of active task dictionaries
    """
    from celery import current_app
     
    inspector = current_app.control.inspect() # type: ignore
    active = inspector.active()
    
    if not active:
        return []
    
    tasks = []
    for worker, task_list in active.items():
        for task in task_list:
            if queue is None or task.get('delivery_info', {}).get('routing_key') == queue:
                tasks.append({
                    'id': task['id'],
                    'name': task['name'],
                    'args': task['args'],
                    'kwargs': task['kwargs'],
                    'worker': worker,
                    'queue': task.get('delivery_info', {}).get('routing_key'),
                })
    
    return tasks


def get_queue_length(queue: str = 'celery') -> int:
    """
    Get number of pending tasks in a queue.
    
    Args:
        queue: Queue name
    
    Returns:
        Number of tasks in queue
    """
    from celery import current_app
    
    with current_app.connection_or_acquire() as conn: # type: ignore
        return conn.default_channel.queue_declare(
            queue=queue, 
            passive=True
        ).message_count


def purge_queue(queue: str) -> int:
    """
    Remove all tasks from a queue.
    
    Args:
        queue: Queue name to purge
    
    Returns:
        Number of tasks purged
    """
    from celery import current_app
    
    return current_app.control.purge() # type: ignore


# Create singleton celery app instance
celery_app = create_celery()

