"""
Celery task for training on platform's GPU workers.
Handles the actual training execution.
"""
from celery import shared_task
from django.utils import timezone
import logging
import traceback
from training.models import TrainingSession
logger = logging.getLogger(__name__)


@shared_task(bind=True, queue='gpu-training', max_retries=3)
def train_model_on_platform_gpu(
    self,
    model_version_id: str,
    training_session_id: str,
    config: dict
):
    """
    Execute model training on platform GPU worker.
    
    This task runs on dedicated GPU workers managed by Celery.
    Uses Ray for distributed training if multiple GPUs available.
    """
    from ml_models.models import ModelVersion
    from training.models import TrainingSession
    import torch
    
    logger.info(f"Starting training for version {model_version_id}")
    
    try:
        # 1. Get model version and training session
        model_version = ModelVersion.objects.select_related(
            'model', 'dataset_version', 'storage_profile'
        ).get(version_id=model_version_id)
        
        training_session = TrainingSession.objects.get(
            session_id=training_session_id
        )
        
        # 2. Update status
        training_session.status = 'initializing'
        training_session.started_at = timezone.now()
        training_session.save()
        
        model_version.status = 'training'
        model_version.save()
        
        # 3. Download dataset from storage
        dataset_path = download_dataset(model_version.dataset_version, model_version.storage_profile)
        
        # 4. Initialize trainer based on framework
        trainer = get_trainer(
            model_version.model.framework.name,
            model_version.model.task.name,
            config
        )
        
        # 5. Setup callbacks for progress tracking
        callbacks = TrainingCallbacks(training_session, self)
        
        # 6. Execute training
        training_session.status = 'running'
        training_session.save()
        
        results = trainer.train(
            dataset_path=dataset_path,
            config=config,
            callbacks=callbacks
        )
        
        # 7. Upload artifacts to storage
        checkpoint_key = upload_checkpoint(
            results['checkpoint_path'],
            model_version,
            model_version.storage_profile
        )
        
        logs_key = upload_logs(
            results['logs_path'],
            model_version,
            model_version.storage_profile
        )
        
        # 8. Update model version with results
        model_version.checkpoint_key = checkpoint_key
        model_version.training_logs_key = logs_key
        model_version.metrics = results['metrics']
        model_version.status = 'trained'
        model_version.save()
        
        # 9. Mark training session complete
        training_session.mark_completed(results['metrics'])
        
        logger.info(f"Training completed for version {model_version_id}")
        
        return {
            'status': 'success',
            'model_version_id': model_version_id,
            'metrics': results['metrics']
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(f"Training failed: {error_msg}")
        logger.error(error_trace)
        
        # Update session and version
        training_session.mark_failed(error_msg, error_trace)
        
        # Retry on transient errors
        if 'CUDA out of memory' in error_msg or 'Connection' in error_msg:
            raise self.retry(exc=e, countdown=300)  # Retry after 5 min
        
        raise


class TrainingCallbacks:
    """Callbacks for tracking training progress"""
    
    def __init__(self, training_session: TrainingSession, celery_task):
        self.training_session = training_session
        self.celery_task = celery_task
    
    def on_epoch_end(self, epoch: int, metrics: dict):
        """Called after each epoch"""
        # Update progress
        self.training_session.update_progress(epoch, metrics)
        
        # Update Celery task state for frontend polling
        self.celery_task.update_state(
            state='PROGRESS',
            meta={
                'current_epoch': epoch,
                'total_epochs': self.training_session.total_epochs,
                'progress': self.training_session.progress,
                'metrics': metrics
            }
        )
    
    def on_batch_end(self, batch: int, total_batches: int, loss: float):
        """Called after each batch (optional, for fine-grained tracking)"""
        pass


def download_dataset(dataset_version, storage_profile):
    """Download dataset from storage to local GPU worker"""
    from storage.services import get_storage_adapter_for_profile
    import tempfile
    import os
    
    adapter = get_storage_adapter_for_profile(storage_profile)
    
    # Create temp directory for dataset
    dataset_dir = tempfile.mkdtemp(prefix='dataset_')
    
    # Download dataset archive
    dataset_key = f"datasets/{dataset_version.id}/data.zip"
    local_path = os.path.join(dataset_dir, 'data.zip')
    
    adapter.download_file(dataset_key, local_path)
    
    # Extract
    import zipfile
    with zipfile.ZipFile(local_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    return dataset_dir


def upload_checkpoint(checkpoint_path: str, model_version, storage_profile):
    """Upload trained checkpoint to storage"""
    from storage.services import get_storage_adapter_for_profile
    
    adapter = get_storage_adapter_for_profile(storage_profile)
    
    # Generate storage key
    storage_key = (
        f"organizations/{model_version.model.organization.org_id}/"
        f"models/{model_version.model.name}/"
        f"v{model_version.version_number}/checkpoint.pt"
    )
    
    # Upload file with metadata
    with open(checkpoint_path, 'rb') as f:
        adapter.upload_file(
            f,
            storage_key,
            content_type='application/octet-stream',
            metadata={
                'organization': model_version.model.organization.slug,
                'model_name': model_version.model.name,
                'version': str(model_version.version_number),
                'type': 'checkpoint'
            }
        )
    
    logger.info(f"Uploaded checkpoint to storage: {storage_key}")
    return storage_key


def upload_logs(logs_path: str, model_version, storage_profile):
    """Upload training logs to storage"""
    from storage.services import get_storage_adapter_for_profile
    
    adapter = get_storage_adapter_for_profile(storage_profile)
    
    storage_key = (
        f"organizations/{model_version.model.organization.org_id}/"
        f"models/{model_version.model.name}/"
        f"v{model_version.version_number}/training.log"
    )
    
    # Upload file with metadata
    with open(logs_path, 'rb') as f:
        adapter.upload_file(
            f,
            storage_key,
            content_type='text/plain',
            metadata={
                'organization': model_version.model.organization.slug,
                'model_name': model_version.model.name,
                'version': str(model_version.version_number),
                'type': 'training_log'
            }
        )
    
    logger.info(f"Uploaded logs to storage: {storage_key}")
    return storage_key


def get_trainer(framework: str, task: str, config: dict):
    """Factory to get appropriate trainer"""
    from training.trainers import (
        YOLOTrainer,
        PyTorchTrainer,
        TensorFlowTrainer
    )
    
    trainers = {
        'yolo': YOLOTrainer,
        'pytorch': PyTorchTrainer,
        'tensorflow': TensorFlowTrainer,
    }
    
    trainer_class = trainers.get(framework.lower())
    if not trainer_class:
        raise ValueError(f"Unsupported framework: {framework}")
    
    return trainer_class(task=task, config=config)


# ============= Worker Configuration =============
"""
Celery worker configuration for GPU workers:

# celeryconfig.py
task_routes = {
    'training.tasks.train_model_on_platform_gpu': {'queue': 'gpu-training'},
}

# Start GPU workers:
celery -A your_project worker -Q gpu-training -c 1 --pool=solo -n gpu-worker-1@%h

# Key points:
# - Use --pool=solo for GPU tasks (avoid process forking with CUDA)
# - Run one worker per GPU: -c 1
# - Each worker handles one training job at a time
# - Use CUDA_VISIBLE_DEVICES to assign specific GPUs to workers

# Start 4 GPU workers (for 4 GPUs):
CUDA_VISIBLE_DEVICES=0 celery -A your_project worker -Q gpu-training -c 1 --pool=solo -n gpu-worker-0@%h &
CUDA_VISIBLE_DEVICES=1 celery -A your_project worker -Q gpu-training -c 1 --pool=solo -n gpu-worker-1@%h &
CUDA_VISIBLE_DEVICES=2 celery -A your_project worker -Q gpu-training -c 1 --pool=solo -n gpu-worker-2@%h &
CUDA_VISIBLE_DEVICES=3 celery -A your_project worker -Q gpu-training -c 1 --pool=solo -n gpu-worker-3@%h &
"""