"""
Celery task for training on platform's GPU workers.
Handles the actual training execution using modular trainer system.
"""
from celery import shared_task
from django.utils import timezone
import logging
import traceback
from pathlib import Path
from common_utils.training.trainers.base import TrainingCallbacks

logger = logging.getLogger(__name__)


@shared_task(bind=True, queue='gpu-training', max_retries=3)
def train_model_on_platform_gpu(
    self,
    job_id: str,
    model_version_id: str,
    training_session_id: str,
    config: dict,
    parent_version_id: str = None
):
    """
    Execute model training on platform GPU worker.
    
    This task runs on dedicated GPU workers managed by Celery.
    Uses the modular trainer system for actual training.
    """
    from ml_models.models import ModelVersion
    from training.models import TrainingSession
    from common_utils.training.trainers.factory import TrainerFactory
    from common_utils.training.trainers.base import TrainingConfig, TrainingCallbacks
    
    logger.info(f"Starting training for job {job_id}")
    
    try:
        # 1. Get model version and training session
        model_version = ModelVersion.objects.select_related(
            'model__framework',
            'model__task',
            'model__organization',
            'dataset_version',
            'storage_profile',
            'parent_version'
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
        logger.info("Downloading dataset")
        dataset_path = download_dataset(
            model_version.dataset_version,
            model_version.storage_profile
        )
        
        # 4. Setup output directory
        output_dir = Path(f"/tmp/training/{job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 5. Prepare training config
        training_config = TrainingConfig(
            dataset_path=dataset_path,
            output_dir=output_dir,
            checkpoint_path=None,  # TODO: Handle parent version
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 16),
            learning_rate=config.get('learning_rate', 0.001),
            image_size=config.get('image_size', 640),
            device='cuda',
            workers=config.get('workers', 4),
            optimizer=config.get('optimizer', 'Adam'),
            weight_decay=config.get('weight_decay', 0.0005),
            scheduler=config.get('scheduler'),
            patience=config.get('patience'),
            augmentation=config.get('augmentation', True),
            model_params=config.get('model_params', {})
        )
        
        # 6. Setup callbacks for progress tracking
        callbacks = PlatformTrainingCallbacks(training_session, self)
        
        # 7. Create trainer
        logger.info(
            f"Creating trainer: framework={model_version.model.framework.name}, "
            f"task={model_version.model.task.name}"
        )
        
        trainer = TrainerFactory.create_trainer(
            framework=model_version.model.framework.name,
            task=model_version.model.task.name,
            config=training_config,
            callbacks=callbacks
        )
        
        # 8. Execute training
        training_session.status = 'running'
        training_session.total_epochs = training_config.epochs
        training_session.save()
        
        logger.info("Starting training execution")
        result = trainer.train()
        
        # 9. Upload artifacts to storage
        logger.info("Uploading training artifacts")
        
        checkpoint_key = upload_checkpoint(
            result.best_checkpoint_path,
            model_version,
            model_version.storage_profile
        )
        
        logs_key = upload_logs(
            result.logs_path,
            model_version,
            model_version.storage_profile
        )
        
        onnx_key = None
        if result.onnx_path and result.onnx_path.exists():
            onnx_key = upload_onnx(
                result.onnx_path,
                model_version,
                model_version.storage_profile
            )
        
        # 10. Update model version with results
        model_version.checkpoint_key = checkpoint_key
        model_version.training_logs_key = logs_key
        if onnx_key:
            model_version.onnx_model_key = onnx_key
        model_version.metrics = result.best_metrics.to_dict()
        model_version.status = 'trained'
        model_version.save()
        
        # 11. Mark training session complete
        training_session.status = 'completed'
        training_session.completed_at = timezone.now()
        training_session.progress = 100.0
        training_session.best_metrics = result.best_metrics.to_dict()
        training_session.current_metrics = result.final_metrics.to_dict()
        training_session.save()
        
        # 12. Cleanup
        cleanup_temp_files(dataset_path, output_dir)
        
        logger.info(f"Training completed successfully for job {job_id}")
        
        return {
            'status': 'success',
            'job_id': job_id,
            'model_version_id': model_version_id,
            'metrics': result.best_metrics.to_dict()
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(f"Training failed for job {job_id}: {error_msg}")
        logger.error(error_trace)
        
        # Update session and version
        training_session.status = 'failed'
        training_session.error_message = error_msg
        training_session.error_traceback = error_trace
        training_session.completed_at = timezone.now()
        training_session.save()
        
        model_version.status = 'failed'
        model_version.error_message = error_msg
        model_version.save()
        
        # Retry on transient errors
        if 'CUDA out of memory' in error_msg or 'Connection' in error_msg:
            raise self.retry(exc=e, countdown=300)  # Retry after 5 min
        
        raise


class PlatformTrainingCallbacks(TrainingCallbacks):
    """Callbacks for tracking training progress on platform"""
    
    def __init__(self, training_session, celery_task):
        self.training_session = training_session
        self.celery_task = celery_task
    
    def on_epoch_end(self, epoch: int, metrics):
        """Called after each epoch"""
        from training.models import TrainingSession
        
        # Update progress in database
        self.training_session.current_epoch = epoch
        self.training_session.progress = (
            (epoch / self.training_session.total_epochs) * 100
            if self.training_session.total_epochs > 0
            else 0
        )
        self.training_session.current_metrics = metrics.to_dict()
        
        # Update best metrics if this is better
        if not self.training_session.best_metrics:
            self.training_session.best_metrics = metrics.to_dict()
        else:
            # Simple logic: lower loss is better
            current_loss = metrics.val_loss or metrics.train_loss
            best_loss = self.training_session.best_metrics.get('val_loss') or \
                       self.training_session.best_metrics.get('train_loss')
            
            if current_loss < best_loss:
                self.training_session.best_metrics = metrics.to_dict()
        
        self.training_session.save()
        
        # Update Celery task state for frontend polling
        self.celery_task.update_state(
            state='PROGRESS',
            meta={
                'current_epoch': epoch,
                'total_epochs': self.training_session.total_epochs,
                'progress': self.training_session.progress,
                'metrics': metrics.to_dict()
            }
        )
        
        logger.info(
            f"Epoch {epoch}/{self.training_session.total_epochs} - "
            f"Loss: {metrics.train_loss:.4f}"
        )


def download_dataset(dataset_version, storage_profile):
    """Download dataset from storage to local GPU worker"""
    from storage.services import get_storage_adapter_for_profile
    import tempfile
    import os
    import zipfile
    
    adapter = get_storage_adapter_for_profile(storage_profile)
    
    # Create temp directory for dataset
    dataset_dir = Path(tempfile.mkdtemp(prefix='dataset_'))
    
    # Download dataset archive
    # Assuming dataset is stored as zip at: datasets/{version_id}/data.zip
    dataset_key = f"datasets/{dataset_version.id}/data.zip"
    local_zip_path = dataset_dir / 'data.zip'
    
    logger.info(f"Downloading dataset from: {dataset_key}")
    
    with open(local_zip_path, 'wb') as f:
        adapter.download_file(dataset_key, f)
    
    # Extract
    logger.info("Extracting dataset")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    # Remove zip file
    local_zip_path.unlink()
    
    logger.info(f"Dataset ready at: {dataset_dir}")
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


def upload_onnx(onnx_path: Path, model_version, storage_profile):
    """Upload ONNX model to storage"""
    from storage.services import get_storage_adapter_for_profile
    
    adapter = get_storage_adapter_for_profile(storage_profile)
    
    storage_key = (
        f"organizations/{model_version.model.organization.org_id}/"
        f"models/{model_version.model.name}/"
        f"v{model_version.version_number}/model.onnx"
    )
    
    # Upload file with metadata
    with open(onnx_path, 'rb') as f:
        adapter.upload_file(
            f,
            storage_key,
            content_type='application/octet-stream',
            metadata={
                'organization': model_version.model.organization.slug,
                'model_name': model_version.model.name,
                'version': str(model_version.version_number),
                'type': 'onnx_model'
            }
        )
    
    logger.info(f"Uploaded ONNX model to storage: {storage_key}")
    return storage_key


def cleanup_temp_files(*paths):
    """Cleanup temporary directories"""
    import shutil
    
    for path in paths:
        if path and Path(path).exists():
            try:
                shutil.rmtree(path)
                logger.info(f"Cleaned up: {path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {path}: {e}")


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