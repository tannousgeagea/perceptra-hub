"""
Training job monitoring service.
Polls provider APIs to sync job status back to database.
"""
import logging
from typing import List
from datetime import timedelta

from django.utils import timezone
from celery import shared_task

from compute.models import TrainingJob
from compute.adapters import get_adapter_for_provider

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Monitors training jobs and syncs status from providers.
    Runs as periodic Celery task.
    """
    
    @staticmethod
    def sync_job_status(training_job: TrainingJob):
        """
        Sync single job status from provider.
        Updates TrainingJob and TrainingSession status.
        """
        if not training_job.external_job_id:
            logger.warning(f"Job {training_job.job_id} has no external ID yet")
            return
        
        try:
            adapter = get_adapter_for_provider(training_job.actual_provider)
            status_info = adapter.get_job_status(training_job.external_job_id)
            
            # Update training session
            session = training_job.training_session
            old_status = session.status
            new_status = status_info['status']
            
            if old_status != new_status:
                logger.info(
                    f"Job {training_job.job_id} status changed: "
                    f"{old_status} → {new_status}"
                )
                
                session.status = new_status
                session.progress = status_info['progress']
                
                if status_info.get('metrics'):
                    session.current_metrics = status_info['metrics']
                
                if new_status == 'failed' and status_info.get('error'):
                    session.error_message = status_info['error']
                
                if new_status in ['completed', 'failed', 'cancelled']:
                    session.completed_at = timezone.now()
                    training_job.completed_at = timezone.now()
                    training_job.save()
                
                session.save()
                
        except Exception as e:
            logger.error(f"Failed to sync job {training_job.job_id}: {e}")
    
    @staticmethod
    def sync_all_active_jobs():
        """Sync all active training jobs"""
        active_jobs = TrainingJob.objects.filter(
            training_session__status__in=['queued', 'initializing', 'running']
        ).select_related('training_session', 'actual_provider')
        
        logger.info(f"Syncing {active_jobs.count()} active training jobs")
        
        for job in active_jobs:
            try:
                TrainingMonitor.sync_job_status(job)
            except Exception as e:
                logger.error(f"Error syncing job {job.job_id}: {e}")
                continue


# ============= Celery Tasks =============

@shared_task(name='training.monitor_jobs')
def monitor_training_jobs():
    """
    Periodic task to monitor all active training jobs.
    Run every 30 seconds via Celery Beat.
    """
    TrainingMonitor.sync_all_active_jobs()


@shared_task(name='training.sync_job')
def sync_training_job(training_job_id: int):
    """
    Sync specific training job status.
    Can be called on-demand from API.
    """
    try:
        job = TrainingJob.objects.select_related(
            'training_session', 'actual_provider'
        ).get(id=training_job_id)
        
        TrainingMonitor.sync_job_status(job)
        return {'status': 'success', 'job_id': job.job_id}
    except TrainingJob.DoesNotExist:
        logger.error(f"Training job {training_job_id} not found")
        return {'status': 'error', 'message': 'Job not found'}


@shared_task(name='training.cleanup_stale_jobs')
def cleanup_stale_jobs():
    """
    Cleanup jobs stuck in queued/running state for too long.
    Run daily via Celery Beat.
    """
    threshold = timezone.now() - timedelta(hours=48)
    
    stale_jobs = TrainingJob.objects.filter(
        training_session__status__in=['queued', 'running'],
        training_session__created_at__lt=threshold
    )
    
    logger.info(f"Found {stale_jobs.count()} stale jobs")
    
    for job in stale_jobs:
        logger.warning(f"Marking stale job as failed: {job.job_id}")
        session = job.training_session
        session.status = 'failed'
        session.error_message = 'Job timed out (exceeded 48 hours)'
        session.completed_at = timezone.now()
        session.save()
        
        job.completed_at = timezone.now()
        job.save()


# ============= Celery Beat Schedule =============
"""
Add to celeryconfig.py or settings.py:

from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    'monitor-training-jobs': {
        'task': 'training.monitor_jobs',
        'schedule': 30.0,  # Every 30 seconds
    },
    'cleanup-stale-jobs': {
        'task': 'training.cleanup_stale_jobs',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
}
"""