from django.db import models
from ml_models.models import ModelVersion
from storage.models import StorageProfile
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _

User = get_user_model()


class TrainingSession(models.Model):
    """
    Represents a single training attempt for a model version.
    Multiple sessions can exist per version (retries, experiments).
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('queued', 'Queued'),
        ('initializing', 'Initializing'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    # Primary identifier
    session_id = models.CharField(
        max_length=255,
        unique=True,
        help_text=_('Unique identifier for this training session')
    )
    
    # Relationships
    model_version = models.ForeignKey(
        ModelVersion,
        on_delete=models.CASCADE,
        related_name='training_sessions',
        help_text=_('Model version being trained')
    )
    
    # Storage (inherit from model_version)
    storage_profile = models.ForeignKey(
        StorageProfile,
        on_delete=models.PROTECT,
        related_name='training_sessions',
        help_text=_('Storage profile for logs and checkpoints')
    )
    
    # Celery task tracking
    task_id = models.CharField(
        max_length=255,
        unique=True,
        null=True,
        blank=True,
        help_text=_('Celery task ID for this training run')
    )
    
    # Status and progress
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )
    progress = models.FloatField(
        default=0.0,
        help_text=_('Training progress percentage (0-100)')
    )
    current_epoch = models.PositiveIntegerField(
        default=0,
        help_text=_('Current training epoch')
    )
    total_epochs = models.PositiveIntegerField(
        default=0,
        help_text=_('Total number of epochs to train')
    )
    
    # Training configuration
    config = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Training configuration for this specific run')
    )
    
    # Real-time metrics
    current_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Current/latest metrics from training')
    )
    best_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Best metrics achieved during training')
    )
    
    # Logging (storage keys, not Django files)
    log_summary = models.TextField(
        blank=True,
        help_text=_('Short summary of training progress/status')
    )
    log_file_key = models.CharField(
        max_length=500,
        blank=True,
        help_text=_('Storage key for detailed log file')
    )
    
    # Error handling
    error_message = models.TextField(
        blank=True,
        help_text=_('Error message if training failed')
    )
    error_traceback = models.TextField(
        blank=True,
        help_text=_('Full error traceback for debugging')
    )
    
    # Resource tracking
    compute_resource = models.CharField(
        max_length=255,
        blank=True,
        help_text=_('Compute resource used (GPU type, worker name, etc.)')
    )
    estimated_time_remaining = models.DurationField(
        null=True,
        blank=True,
        help_text=_('Estimated time until completion')
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text=_('When training actually started')
    )
    completed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text=_('When training finished (success or failure)')
    )
    updated_at = models.DateTimeField(auto_now=True)
    
    # Audit
    triggered_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='triggered_training_sessions'
    )

    class Meta:
        db_table = "training_session"
        verbose_name_plural = "Training Sessions"
        indexes = [
            models.Index(fields=['model_version', 'created_at']),
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['task_id']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"Training {self.model_version} - {self.status}"
    
    @property
    def duration(self):
        """Calculate training duration"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if training is currently active"""
        return self.status in ['pending', 'queued', 'initializing', 'running']
    
    @property
    def is_finished(self) -> bool:
        """Check if training is finished (regardless of outcome)"""
        return self.status in ['completed', 'failed', 'cancelled']
    
    def get_logs_url(self, expiration: int = 3600) -> str:
        """Generate presigned URL for logs download"""
        from storage.services import get_storage_adapter_for_profile
        
        if not self.log_file_key:
            return None
        
        if self.storage_profile.backend == "local":
            return f"http://localhost:81/{self.storage_profile.config['base_path']}/{self.log_file_key}"
        
        adapter = get_storage_adapter_for_profile(self.storage_profile)
        presigned = adapter.generate_presigned_url(
            self.log_file_key,
            expiration=expiration,
            method='GET'
        )
        return presigned.url
    
    def update_progress(self, epoch: int, metrics: dict = None):
        """Update training progress"""
        self.current_epoch = epoch
        if self.total_epochs > 0:
            self.progress = (epoch / self.total_epochs) * 100
        
        if metrics:
            self.current_metrics = metrics
            
            # Update best metrics
            if not self.best_metrics:
                self.best_metrics = metrics
            else:
                # Example: update if current is better (customize per metric)
                for key, value in metrics.items():
                    if key.startswith('loss'):
                        # Lower is better for loss
                        if value < self.best_metrics.get(key, float('inf')):
                            self.best_metrics[key] = value
                    else:
                        # Higher is better for accuracy, mAP, etc.
                        if value > self.best_metrics.get(key, 0):
                            self.best_metrics[key] = value
        
        self.save()
    
    def mark_failed(self, error_message: str, traceback: str = None):
        """Mark training as failed"""
        from django.utils import timezone
        
        self.status = 'failed'
        self.error_message = error_message
        if traceback:
            self.error_traceback = traceback
        self.completed_at = timezone.now()
        self.save()
        
        # Also update model version status
        self.model_version.status = 'failed'
        self.model_version.error_message = error_message
        self.model_version.save()
    
    def mark_completed(self, final_metrics: dict = None):
        """Mark training as completed successfully"""
        from django.utils import timezone
        
        self.status = 'completed'
        self.progress = 100.0
        self.completed_at = timezone.now()
        
        if final_metrics:
            self.current_metrics = final_metrics
            self.best_metrics = final_metrics
        
        self.save()
        
        # Update model version status
        self.model_version.status = 'trained'
        if final_metrics:
            self.model_version.metrics = final_metrics
        self.model_version.save()


class TrainingCheckpoint(models.Model):
    """
    Stores intermediate checkpoints during training.
    Useful for resuming training or analyzing training progression.
    """
    training_session = models.ForeignKey(
        TrainingSession,
        on_delete=models.CASCADE,
        related_name='checkpoints'
    )
    
    epoch = models.PositiveIntegerField(
        help_text=_('Epoch number when checkpoint was saved')
    )
    
    # Storage key instead of file path
    checkpoint_key = models.CharField(
        max_length=500,
        help_text=_('Storage key for checkpoint file')
    )
    
    metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text=_('Metrics at this checkpoint')
    )
    
    is_best = models.BooleanField(
        default=False,
        help_text=_('Whether this was the best checkpoint')
    )
    
    file_size_bytes = models.BigIntegerField(
        null=True,
        blank=True,
        help_text=_('Size of checkpoint file')
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "training_checkpoint"
        verbose_name_plural = "Training Checkpoints"
        unique_together = [('training_session', 'epoch')]
        indexes = [
            models.Index(fields=['training_session', 'epoch']),
            models.Index(fields=['is_best']),
        ]
        ordering = ['epoch']
    
    def __str__(self):
        return f"Checkpoint {self.training_session} - Epoch {self.epoch}"
    
    def get_checkpoint_url(self, expiration: int = 3600) -> str:
        """Generate presigned URL for checkpoint download"""
        from storage.services import get_storage_adapter_for_profile
        
        storage_profile = self.training_session.storage_profile
        
        if storage_profile.backend == "local":
            return f"http://localhost:81/{storage_profile.config['base_path']}/{self.checkpoint_key}"
        
        adapter = get_storage_adapter_for_profile(storage_profile)
        presigned = adapter.generate_presigned_url(
            self.checkpoint_key,
            expiration=expiration,
            method='GET'
        )
        return presigned.url