"""
Base trainer interface - framework agnostic.
All trainers must implement this interface.

Location: training/trainers/base.py
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Universal training configuration"""
    # Paths
    dataset_path: Path
    output_dir: Path
    checkpoint_path: Optional[Path] = None
    
    # Training params
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    image_size: int = 640
    
    # Hardware
    device: str = 'cuda'  # cuda, cpu, mps
    workers: int = 4
    
    # Optimizer
    optimizer: str = 'Adam'
    weight_decay: float = 0.0005
    momentum: float = 0.9
    
    # Scheduler
    scheduler: Optional[str] = None  # cosine, step, etc.
    
    # Early stopping
    patience: Optional[int] = None
    
    # Augmentation
    augmentation: bool = True
    
    # Model specific (passed through)
    model_params: Dict[str, Any] = None
    
    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
        if self.checkpoint_path:
            self.checkpoint_path = Path(self.checkpoint_path)
        if self.model_params is None:
            self.model_params = {}


@dataclass
class TrainingMetrics:
    """Training metrics output"""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    
    # Task-specific metrics
    metrics: Dict[str, float] = None
    
    # Learning rate
    learning_rate: float = 0.0
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'learning_rate': self.learning_rate,
            **self.metrics
        }


@dataclass
class TrainingResult:
    """Final training result"""
    success: bool
    best_checkpoint_path: Path
    final_checkpoint_path: Path
    logs_path: Path
    
    # Best metrics achieved
    best_metrics: TrainingMetrics
    final_metrics: TrainingMetrics
    
    # Training info
    total_epochs: int
    training_time_seconds: float
    
    # Optional exported models
    onnx_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'best_checkpoint_path': str(self.best_checkpoint_path),
            'final_checkpoint_path': str(self.final_checkpoint_path),
            'logs_path': str(self.logs_path),
            'best_metrics': self.best_metrics.to_dict(),
            'final_metrics': self.final_metrics.to_dict(),
            'total_epochs': self.total_epochs,
            'training_time_seconds': self.training_time_seconds,
            'onnx_path': str(self.onnx_path) if self.onnx_path else None
        }


class TrainingCallbacks:
    """Callbacks for training progress"""
    
    def on_train_start(self, config: TrainingConfig):
        """Called at start of training"""
        pass
    
    def on_epoch_start(self, epoch: int):
        """Called at start of each epoch"""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics):
        """Called at end of each epoch"""
        pass
    
    def on_batch_end(self, batch: int, total_batches: int, loss: float):
        """Called after each batch"""
        pass
    
    def on_train_end(self, result: TrainingResult):
        """Called at end of training"""
        pass
    
    def on_checkpoint_saved(self, epoch: int, path: Path, is_best: bool):
        """Called when checkpoint is saved"""
        pass


class BaseTrainer(ABC):
    """
    Abstract base trainer class.
    All framework-specific trainers inherit from this.
    """
    
    def __init__(
        self,
        task: str,
        config: TrainingConfig,
        callbacks: Optional[TrainingCallbacks] = None
    ):
        """
        Initialize trainer.
        
        Args:
            task: Task type (object-detection, classification, segmentation)
            config: Training configuration
            callbacks: Optional callbacks for progress tracking
        """
        self.task = task
        self.config = config
        self.callbacks = callbacks or TrainingCallbacks()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Initialized {self.__class__.__name__} for task: {task}")
    
    def _setup_logging(self):
        """Setup logging to file"""
        log_file = self.config.output_dir / 'training.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        logger.addHandler(file_handler)
    
    @abstractmethod
    def prepare_dataset(self) -> Any:
        """
        Prepare dataset for training.
        Must be implemented by subclass.
        
        Returns:
            Dataset object(s) ready for training
        """
        pass
    
    @abstractmethod
    def create_model(self) -> Any:
        """
        Create and return model instance.
        Must be implemented by subclass.
        
        Returns:
            Model object ready for training
        """
        pass
    
    @abstractmethod
    def train_epoch(self, epoch: int) -> TrainingMetrics:
        """
        Train single epoch.
        Must be implemented by subclass.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Metrics for this epoch
        """
        pass
    
    @abstractmethod
    def validate(self, epoch: int) -> TrainingMetrics:
        """
        Run validation.
        Must be implemented by subclass.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Validation metrics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, epoch: int, is_best: bool) -> Path:
        """
        Save model checkpoint.
        Must be implemented by subclass.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        
        Returns:
            Path to saved checkpoint
        """
        pass
    
    @abstractmethod
    def export_model(self, checkpoint_path: Path) -> Optional[Path]:
        """
        Export model to deployment format (ONNX, TorchScript, etc.).
        Must be implemented by subclass.
        
        Args:
            checkpoint_path: Path to checkpoint to export
        
        Returns:
            Path to exported model, or None if export not supported
        """
        pass
    
    def train(self) -> TrainingResult:
        """
        Main training loop.
        This is the same for all trainers - delegates to abstract methods.
        
        Returns:
            TrainingResult with paths and metrics
        """
        import time
        
        start_time = time.time()
        
        # Callback: training start
        self.callbacks.on_train_start(self.config)
        logger.info("Starting training")
        
        # Prepare dataset
        logger.info("Preparing dataset")
        self.prepare_dataset()
        
        # Create model
        logger.info("Creating model")
        self.create_model()
        
        # Training loop
        best_metrics = None
        best_metric_value = float('inf')  # For loss-based (lower is better)
        best_checkpoint_path = None
        
        for epoch in range(1, self.config.epochs + 1):
            self.callbacks.on_epoch_start(epoch)
            logger.info(f"Epoch {epoch}/{self.config.epochs}")
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train loss: {train_metrics.train_loss:.4f}")
            
            # Validate
            val_metrics = self.validate(epoch)
            if val_metrics:
                logger.info(f"Val loss: {val_metrics.val_loss:.4f}")
                train_metrics.val_loss = val_metrics.val_loss
                train_metrics.metrics.update(val_metrics.metrics)
            
            # Check if best model
            metric_value = val_metrics.val_loss if val_metrics else train_metrics.train_loss
            is_best = metric_value < best_metric_value
            
            if is_best:
                best_metric_value = metric_value
                best_metrics = train_metrics
                logger.info(f"New best model! Metric: {metric_value:.4f}")
            
            # Save checkpoint
            checkpoint_path = self.save_checkpoint(epoch, is_best)
            if is_best:
                best_checkpoint_path = checkpoint_path
            
            self.callbacks.on_checkpoint_saved(epoch, checkpoint_path, is_best)
            self.callbacks.on_epoch_end(epoch, train_metrics)
            
            # Early stopping
            if self.config.patience and epoch > self.config.patience:
                # Simple early stopping: check if no improvement in last N epochs
                # (Implement more sophisticated logic in subclasses if needed)
                pass
        
        # Training complete
        training_time = time.time() - start_time
        logger.info(f"Training complete in {training_time:.2f}s")
        
        # Export model
        onnx_path = None
        if best_checkpoint_path:
            logger.info("Exporting model")
            onnx_path = self.export_model(best_checkpoint_path)
        
        # Get final checkpoint
        final_checkpoint_path = self.config.output_dir / f'checkpoint_final.pt'
        
        # Create result
        result = TrainingResult(
            success=True,
            best_checkpoint_path=best_checkpoint_path,
            final_checkpoint_path=final_checkpoint_path,
            logs_path=self.config.output_dir / 'training.log',
            best_metrics=best_metrics,
            final_metrics=train_metrics,
            total_epochs=self.config.epochs,
            training_time_seconds=training_time,
            onnx_path=onnx_path
        )
        
        self.callbacks.on_train_end(result)
        
        return result
    
    @staticmethod
    def get_device(device_name: str = 'cuda'):
        """Get torch device"""
        import torch
        
        if device_name == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device_name == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')