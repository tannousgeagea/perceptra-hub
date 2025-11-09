"""
Celery tasks for dataset export with monitoring and retry logic.
"""
from celery import shared_task, Task
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_prerun, task_postrun, task_failure
import logging

from common_utils.dataset_export.streaming.manager import StreamingDatasetExportManager
from projects.models import Version
from django.utils import timezone

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Base task with callbacks and error handling."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        version_id = args[0] if args else None
        logger.error(
            f"Export task failed. "
            f"Task: {task_id}, "
            f"Version: {version_id}, "
            f"Error: {exc}"
        )
        
        # Update version status
        if version_id:
            try:
                version = Version.objects.get(id=version_id)
                version.export_status = 'failed'
                version.error_log = str(exc)[:1000]
                version.save(update_fields=['export_status', 'error_log'])
            except Exception as e:
                logger.error(f"Failed to update version status: {e}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        version_id = args[0] if args else None
        logger.warning(
            f"Export task retrying. "
            f"Task: {task_id}, "
            f"Version: {version_id}, "
            f"Attempt: {self.request.retries + 1}/{self.max_retries}"
        )


@shared_task(
    bind=True,
    base=CallbackTask,
    name='dataset.export_version',
    max_retries=3,
    soft_time_limit=3600,  # 1 hour
    time_limit=3900,  # 65 minutes (hard limit)
    acks_late=True,
    reject_on_worker_lost=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,  # Max 10 minutes
    retry_jitter=True
)
def export_dataset_task(self, version_id: int):
    """
    Export dataset version (Celery task).
    
    Args:
        version_id: Version primary key
        
    Returns:
        bool: Success status
    """
    try:
        logger.info(f"Starting export task for version {version_id}")
        
        # Run export
        success = StreamingDatasetExportManager.export_version(version_id)
        
        if not success:
            raise Exception("Export failed - check logs for details")
        
        logger.info(f"Export task completed for version {version_id}")
        return True
        
    except SoftTimeLimitExceeded:
        logger.error(f"Export task exceeded time limit for version {version_id}")
        
        # Update version
        try:
            version = Version.objects.get(id=version_id)
            version.export_status = 'failed'
            version.error_log = 'Export exceeded time limit (1 hour)'
            version.save(update_fields=['export_status', 'error_log'])
        except:
            pass
        
        raise
    
    except Exception as exc:
        logger.exception(f"Export task error for version {version_id}: {exc}")
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        
        raise


@shared_task(name='dataset.cleanup_failed_exports')
def cleanup_failed_exports():
    """
    Cleanup task for stuck or failed exports.
    
    Runs periodically to reset exports stuck in 'processing' state.
    """
    from django.utils import timezone
    from datetime import timedelta
    
    # Find versions stuck in processing for > 2 hours
    cutoff_time = timezone.now() - timedelta(hours=2)
    
    stuck_versions = Version.objects.filter(
        export_status='processing',
        updated_at__lt=cutoff_time
    )
    
    count = stuck_versions.count()
    
    if count > 0:
        logger.warning(f"Found {count} stuck export(s), resetting to failed")
        
        stuck_versions.update(
            export_status='failed',
            error_log='Export stuck in processing state - automatically reset'
        )
    
    return count


@shared_task(name='dataset.generate_export_metrics')
def generate_export_metrics():
    """
    Generate metrics for monitoring.
    
    Tracks export success rate, average time, etc.
    """
    from django.db.models import Count, Q
    from datetime import timedelta
    
    cutoff = timezone.now() - timedelta(days=7)
    
    stats = Version.objects.filter(
        created_at__gte=cutoff
    ).aggregate(
        total=Count('id'),
        completed=Count('id', filter=Q(export_status='completed')),
        failed=Count('id', filter=Q(export_status='failed')),
        processing=Count('id', filter=Q(export_status='processing'))
    )
    
    logger.info(
        f"Export metrics (last 7 days): "
        f"Total={stats['total']}, "
        f"Completed={stats['completed']}, "
        f"Failed={stats['failed']}, "
        f"Processing={stats['processing']}"
    )
    
    return stats


# ============= Task Monitoring =============

@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **extra):
    """Log when task starts."""
    if task.name.startswith('dataset.'):
        logger.info(f"Task started: {task.name} ({task_id})")


@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, **extra):
    """Log when task completes."""
    if task.name.startswith('dataset.'):
        logger.info(f"Task completed: {task.name} ({task_id})")


@task_failure.connect
def task_failure_handler(task_id, exception, args, kwargs, traceback, **extra):
    """Log when task fails."""
    logger.error(
        f"Task failed: {task_id}, "
        f"Exception: {exception}"
    )