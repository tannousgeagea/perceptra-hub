"""
YOLO trainer implementation using Ultralytics YOLO.
Supports YOLOv8, YOLOv9, YOLOv10, YOLO11.

Location: training/trainers/yolo_trainer.py
"""
from pathlib import Path
from typing import Any, Optional
import logging

from .base import BaseTrainer, TrainingConfig, TrainingMetrics, TrainingResult
import torch

logger = logging.getLogger(__name__)


class YOLOTrainer(BaseTrainer):
    """
    Trainer for YOLO models (object detection, segmentation, classification).
    Uses Ultralytics YOLO library.
    """
    
    def __init__(self, task: str, config: TrainingConfig, callbacks=None):
        super().__init__(task, config, callbacks)

        self.model = None
        self.yolo_task = self._map_task_to_yolo(task)

        # YOLO-specific config
        self.model_size = config.model_params.get('model_size', 'n')  # n, s, m, l, x
        self.model_version = config.model_params.get('version', '11')  # 8, 9, 10, 11
        self.pretrained = config.model_params.get('pretrained', True)

        # Resolve actual available device, falling back to CPU if requested device unavailable
        self.config.device = self._resolve_device(config.device)
    
    @staticmethod
    def _resolve_device(device) -> str:
        """Return the best available device, falling back to CPU if requested device is absent."""
        import torch
        device = str(device)
        # Treat numeric strings and 'cuda' as CUDA requests
        if device == 'cuda' or device.isdigit():
            if torch.cuda.is_available():
                return device
            logger.warning(f"CUDA not available (requested '{device}'), falling back to CPU")
            return 'cpu'
        if device == 'mps':
            if torch.backends.mps.is_available():
                return 'mps'
            logger.warning("MPS not available, falling back to CPU")
            return 'cpu'
        return device  # 'cpu' or anything explicit passes through

    def _map_task_to_yolo(self, task: str) -> str:
        """Map generic task to YOLO task"""
        mapping = {
            'object-detection': 'detect',
            'segmentation': 'segment',
            'classification': 'classify',
            'instance-segmentation': 'segment',
            'pose-estimation': 'pose',
            'obb': 'obb'  # Oriented bounding boxes
        }
        return mapping.get(task, 'detect')
    
    def _get_model_name(self) -> str:
        """Get YOLO model name"""
        # Format: yolo11n.pt, yolov8s-seg.pt, etc.
        version_map = {
            '8': 'yolov8',
            '9': 'yolov9',
            '10': 'yolov10',
            '11': 'yolo11'
        }
        
        version_name = version_map.get(self.model_version, 'yolo11')
        
        # Add task suffix for non-detection tasks
        if self.yolo_task == 'segment':
            suffix = '-seg'
        elif self.yolo_task == 'classify':
            suffix = '-cls'
        elif self.yolo_task == 'pose':
            suffix = '-pose'
        elif self.yolo_task == 'obb':
            suffix = '-obb'
        else:
            suffix = ''
        
        return f"{version_name}{self.model_size}{suffix}.pt"
    
    def prepare_dataset(self):
        """
        Prepare YOLO dataset.
        Expects dataset in YOLO format with data.yaml
        """
        # Verify dataset structure
        dataset_path = self.config.dataset_path
        data_yaml = dataset_path / 'data.yaml'
        
        if not data_yaml.exists():
            raise FileNotFoundError(
                f"YOLO dataset requires data.yaml at {data_yaml}\n"
                f"Expected structure:\n"
                f"  {dataset_path}/\n"
                f"    data.yaml\n"
                f"    train/images/\n"
                f"    train/labels/\n"
                f"    val/images/\n"
                f"    val/labels/"
            )

        # Write a derived copy to output_dir with the resolved path field.
        # Never modify the original data.yaml so it survives interruptions.
        import yaml
        with open(data_yaml) as f:
            data = yaml.safe_load(f)
        data['path'] = str(dataset_path.resolve())
        derived_yaml = self.config.output_dir / 'data_training.yaml'
        with open(derived_yaml, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"data_training.yaml written to: {derived_yaml}")

        self.data_yaml_path = derived_yaml
        logger.info(f"Dataset prepared: {data_yaml} → {derived_yaml}")

        return derived_yaml
    
    def create_model(self):
        """Create YOLO model"""
        from ultralytics import YOLO
        
        model_name = self._get_model_name()
        
        # Load model
        if self.config.checkpoint_path and self.config.checkpoint_path.exists():
            # Resume from checkpoint
            logger.info(f"Loading checkpoint: {self.config.checkpoint_path}")
            self.model = YOLO(str(self.config.checkpoint_path))
        elif self.pretrained:
            # Load pretrained model
            logger.info(f"Loading pretrained model: {model_name}")
            self.model = YOLO(model_name)
        else:
            # Create from scratch
            logger.info(f"Creating model from scratch: {model_name}")
            # Load architecture config without weights
            config_name = model_name.replace('.pt', '.yaml')
            self.model = YOLO(config_name)
        
        logger.info(f"Model created: {model_name}")
        return self.model
    
    def train_epoch(self, epoch: int) -> TrainingMetrics:
        """
        Train using YOLO's built-in training.
        Note: YOLO handles the full training loop, so this is called once.
        """
        # YOLO trains for all epochs at once, so we only run this on epoch 1
        if epoch > 1:
            # Return cached metrics for subsequent epochs
            return self._get_cached_metrics(epoch)
        
        logger.info("Starting YOLO training")
        
        # Prepare training arguments
        train_args = {
            'data': str(self.data_yaml_path),
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': self.config.image_size,
            'lr0': self.config.learning_rate,
            'device': self.config.device,
            'workers': self.config.workers,
            'project': str(self.config.output_dir),
            'name': 'train',
            'exist_ok': True,
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'patience': self.config.patience or 50,
            'verbose': True,
            'seed': 42,
        }
        
        # Add optimizer settings
        if self.config.optimizer:
            train_args['optimizer'] = self.config.optimizer
        
        # Add augmentation settings
        if not self.config.augmentation:
            train_args['augment'] = False
        
        # Add model-specific params
        train_args.update(self.config.model_params.get('train_params', {}))
        
        # Register per-epoch progress callback into Ultralytics' event system
        def _on_epoch_end(yolo_trainer_obj):
            import math
            epoch = yolo_trainer_obj.epoch + 1  # ultralytics is 0-indexed
            total = yolo_trainer_obj.epochs
            raw = {
                'train_loss': float(yolo_trainer_obj.loss) if hasattr(yolo_trainer_obj, 'loss') else 0.0,
            }
            if hasattr(yolo_trainer_obj, 'metrics') and yolo_trainer_obj.metrics:
                raw.update({k: float(v) for k, v in yolo_trainer_obj.metrics.items()
                            if isinstance(v, (int, float)) and math.isfinite(float(v))})
            self._call_progress(epoch, total, raw)
            # Also drive the standard epoch callback so DB is updated each epoch
            epoch_metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=raw.get('train_loss', 0.0),
                metrics={k: v for k, v in raw.items() if k != 'train_loss'},
                learning_rate=self.config.learning_rate,
            )
            self.callbacks.on_epoch_end(epoch, epoch_metrics)

        self.model.add_callback('on_train_epoch_end', _on_epoch_end)

        # Train model
        results = self.model.train(**train_args)
        
        # Store results for later access
        self.training_results = results
        
        # Extract metrics from final epoch
        metrics = self._extract_metrics_from_results(results, self.config.epochs)
        
        return metrics
    
    def validate(self, epoch: int) -> Optional[TrainingMetrics]:
        """
        Validate model.
        YOLO handles validation during training, so we extract metrics.
        """
        if epoch == 1 and hasattr(self, 'training_results'):
            # Validation is done during training
            return None  # Metrics already included in train_epoch
        
        return None
    
    def save_checkpoint(self, epoch: int, is_best: bool) -> Path:
        """
        Save checkpoint.
        YOLO saves automatically, we just return the path.
        """
        # YOLO saves to: output_dir/train/weights/
        weights_dir = self.config.output_dir / 'train' / 'weights'
        
        if is_best:
            checkpoint_path = weights_dir / 'best.pt'
        else:
            checkpoint_path = weights_dir / 'last.pt'
        
        if checkpoint_path.exists():
            logger.info(f"Checkpoint: {checkpoint_path}")
            return checkpoint_path
        else:
            # Fallback if YOLO structure is different
            logger.warning(f"Expected checkpoint not found: {checkpoint_path}")
            return self.config.output_dir / f'checkpoint_epoch_{epoch}.pt'
    
    def export_model(self, checkpoint_path: Path) -> Optional[Path]:
        """Export model to ONNX"""
        try:
            from ultralytics import YOLO
            
            logger.info(f"Exporting model to ONNX: {checkpoint_path}")
            
            model = YOLO(str(checkpoint_path))
            
            # Export to ONNX
            onnx_path = model.export(
                format='onnx',
                imgsz=self.config.image_size,
                simplify=True,
                opset=12
            )
            
            logger.info(f"Model exported to: {onnx_path}")
            return Path(onnx_path)
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            return None
    
    def _extract_metrics_from_results(self, results, epoch: int) -> TrainingMetrics:
        """Extract metrics from YOLO results object"""
        import math

        try:
            metrics_dict = {}

            if hasattr(results, 'box'):
                if hasattr(results.box, 'map50'):
                    metrics_dict['mAP50'] = float(results.box.map50)
                elif hasattr(results.box, 'map'):
                    metrics_dict['mAP50'] = float(results.box.map)
                if hasattr(results.box, 'map75'):
                    metrics_dict['mAP75'] = float(results.box.map75)

            train_loss = 0.0
            val_loss = None

            if hasattr(results, 'results_dict'):
                rd = results.results_dict
                train_loss = rd.get('train/loss', 0.0)
                val_loss = rd.get('val/loss')
                if 'metrics/precision(B)' in rd:
                    metrics_dict['precision'] = rd['metrics/precision(B)']
                if 'metrics/recall(B)' in rd:
                    metrics_dict['recall'] = rd['metrics/recall(B)']

            # Drop NaN/Inf values that can't be serialized
            metrics_dict = {
                k: v for k, v in metrics_dict.items()
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            }
            if train_loss and not math.isfinite(float(train_loss)):
                logger.warning(f"Non-finite train_loss at epoch {epoch}, resetting to 0.0")
                train_loss = 0.0

            return TrainingMetrics(
                epoch=epoch,
                train_loss=float(train_loss),
                val_loss=float(val_loss) if val_loss is not None else None,
                metrics=metrics_dict,
                learning_rate=self.config.learning_rate
            )

        except Exception as e:
            logger.warning(f"Could not extract detailed metrics: {e}")
            return TrainingMetrics(epoch=epoch, train_loss=0.0, metrics={})
    
    def _get_cached_metrics(self, epoch: int) -> TrainingMetrics:
        """Get metrics for a specific epoch from cached results"""
        # This is a placeholder - in practice, YOLO trains all epochs at once
        # So we don't call train_epoch multiple times
        return TrainingMetrics(
            epoch=epoch,
            train_loss=0.0,
            metrics={}
        )
    
    def train(self) -> TrainingResult:
        """
        Override train to handle YOLO's all-at-once training.
        """
        import time

        start_time = time.time()
        self.callbacks.on_train_start(self.config)
        logger.info("Starting YOLO training")

        try:
            self.prepare_dataset()
            self.create_model()

            # YOLO runs all epochs internally; _on_epoch_end fires per epoch via callback
            train_metrics = self.train_epoch(1)
        except Exception as e:
            self.callbacks.on_train_error(self.config.epochs, e)
            raise

        # Get checkpoint paths
        weights_dir = self.config.output_dir / 'train' / 'weights'
        best_checkpoint = weights_dir / 'best.pt'
        final_checkpoint = weights_dir / 'last.pt'

        # Validate checkpoint exists
        if not best_checkpoint.exists():
            logger.warning(f"Expected best checkpoint not found: {best_checkpoint}")
            best_checkpoint = final_checkpoint if final_checkpoint.exists() else best_checkpoint

        onnx_path = self.export_model(best_checkpoint)
        training_time = time.time() - start_time

        result = TrainingResult(
            success=True,
            best_checkpoint_path=best_checkpoint,
            final_checkpoint_path=final_checkpoint,
            logs_path=self.config.output_dir / 'train' / 'results.csv',
            best_metrics=train_metrics,
            final_metrics=train_metrics,
            total_epochs=self.config.epochs,
            training_time_seconds=training_time,
            onnx_path=onnx_path
        )

        self.callbacks.on_train_end(result)
        return result