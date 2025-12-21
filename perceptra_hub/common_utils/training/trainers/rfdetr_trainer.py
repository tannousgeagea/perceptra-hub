"""
Roboflow DETR (RT-DETR) trainer implementation.
Uses the Roboflow implementation of RT-DETR.

Location: training/trainers/rfdetr_trainer.py
"""
from pathlib import Path
from typing import Any, Optional, Dict
import logging
import json

from .base import BaseTrainer, TrainingConfig, TrainingMetrics, TrainingResult
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class RFDETRTrainer(BaseTrainer):
    """
    Trainer for Roboflow DETR (RT-DETR).
    Implements custom training loop with PyTorch.
    """
    
    def __init__(self, task: str, config: TrainingConfig, callbacks=None):
        super().__init__(task, config, callbacks)
        
        if task != 'object-detection':
            raise ValueError(f"RF-DETR only supports object-detection, got: {task}")
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision
        
        # RF-DETR specific config
        self.model_variant = config.model_params.get('variant', 'rtdetr-l')  # rtdetr-l, rtdetr-x
        self.num_classes = config.model_params.get('num_classes')
        self.pretrained_weights = config.model_params.get('pretrained_weights')
        self.use_amp = config.model_params.get('use_amp', True)  # Automatic Mixed Precision
        
        self.device = self.get_device(config.device)
        
    def prepare_dataset(self):
        """
        Prepare COCO format dataset for RT-DETR.
        Expects COCO annotations.
        """
        from training.trainers.datasets.coco_dataset import COCODetectionDataset
        from training.trainers.datasets.transforms import get_transforms
        
        dataset_path = self.config.dataset_path
        
        # Paths to COCO format annotations
        train_ann = dataset_path / 'annotations' / 'instances_train.json'
        val_ann = dataset_path / 'annotations' / 'instances_val.json'
        train_img_dir = dataset_path / 'train'
        val_img_dir = dataset_path / 'val'
        
        if not train_ann.exists():
            raise FileNotFoundError(
                f"COCO annotations not found: {train_ann}\n"
                f"Expected structure:\n"
                f"  {dataset_path}/\n"
                f"    annotations/\n"
                f"      instances_train.json\n"
                f"      instances_val.json\n"
                f"    train/  (images)\n"
                f"    val/    (images)"
            )
        
        # Get number of classes from annotations if not specified
        if self.num_classes is None:
            with open(train_ann) as f:
                coco_data = json.load(f)
                self.num_classes = len(coco_data['categories'])
                logger.info(f"Detected {self.num_classes} classes from dataset")
        
        # Create datasets
        train_transforms = get_transforms(
            image_size=self.config.image_size,
            is_train=True,
            augment=self.config.augmentation
        )
        val_transforms = get_transforms(
            image_size=self.config.image_size,
            is_train=False,
            augment=False
        )
        
        train_dataset = COCODetectionDataset(
            img_dir=train_img_dir,
            ann_file=train_ann,
            transforms=train_transforms
        )
        
        val_dataset = COCODetectionDataset(
            img_dir=val_img_dir,
            ann_file=val_ann,
            transforms=val_transforms
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        logger.info(
            f"Dataset prepared: {len(train_dataset)} train, "
            f"{len(val_dataset)} val images"
        )
        
        return train_dataset, val_dataset
    
    def _collate_fn(self, batch):
        """Custom collate function for DETR"""
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        # Stack images
        images = torch.stack(images, dim=0)
        
        return images, targets
    
    def create_model(self):
        """Create RT-DETR model"""
        try:
            # Try to import RT-DETR from official repo or roboflow
            from rtdetr_pytorch import RTDETR
            
            logger.info(f"Creating RT-DETR model: {self.model_variant}")
            
            # Initialize model
            self.model = RTDETR(
                variant=self.model_variant,
                num_classes=self.num_classes,
                pretrained=self.pretrained_weights is not None
            )
            
            # Load pretrained weights if specified
            if self.pretrained_weights:
                logger.info(f"Loading pretrained weights: {self.pretrained_weights}")
                checkpoint = torch.load(self.pretrained_weights, map_location='cpu')
                self.model.load_state_dict(checkpoint['model'], strict=False)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Setup optimizer
            self._setup_optimizer()
            
            # Setup mixed precision scaler
            if self.use_amp:
                self.scaler = torch.cuda.amp.GradScaler()
                logger.info("Using Automatic Mixed Precision (AMP)")
            
            logger.info(f"Model created with {self._count_parameters():,} parameters")
            
            return self.model
            
        except ImportError:
            raise ImportError(
                "RT-DETR not installed. Install with:\n"
                "pip install rtdetr-pytorch\n"
                "or clone from: https://github.com/lyuwenyu/RT-DETR"
            )
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Separate parameters for backbone and transformer
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'backbone' not in n and p.requires_grad],
                'lr': self.config.learning_rate
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'backbone' in n and p.requires_grad],
                'lr': self.config.learning_rate * 0.1  # Lower LR for backbone
            }
        ]
        
        # Create optimizer
        if self.config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                param_groups,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                param_groups,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Create scheduler
        if self.config.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        
        logger.info(f"Optimizer: {self.config.optimizer}, Scheduler: {self.config.scheduler}")
    
    def train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train single epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            
            # Move targets to device
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Forward pass with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss_dict = self.model(images, targets)
                    loss = sum(loss_dict.values())
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Callback: batch end
            if batch_idx % 10 == 0:
                self.callbacks.on_batch_end(batch_idx, num_batches, loss.item())
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Step scheduler
        if self.scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return TrainingMetrics(
            epoch=epoch,
            train_loss=avg_loss,
            learning_rate=current_lr,
            metrics={}
        )
    
    def validate(self, epoch: int) -> Optional[TrainingMetrics]:
        """Run validation and compute mAP"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        # Collect predictions for mAP calculation
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # Get loss
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())
                total_loss += loss.item()
                
                # Get predictions for mAP
                predictions = self.model(images)  # Inference mode
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        avg_loss = total_loss / num_batches
        
        # Calculate mAP
        metrics_dict = self._calculate_map(all_predictions, all_targets)
        
        logger.info(
            f"Validation - Loss: {avg_loss:.4f}, "
            f"mAP50: {metrics_dict.get('mAP50', 0.0):.4f}"
        )
        
        return TrainingMetrics(
            epoch=epoch,
            train_loss=0.0,
            val_loss=avg_loss,
            metrics=metrics_dict
        )
    
    def _calculate_map(self, predictions, targets) -> Dict[str, float]:
        """Calculate mAP metrics"""
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
            
            metric = MeanAveragePrecision()
            metric.update(predictions, targets)
            results = metric.compute()
            
            return {
                'mAP': float(results['map'].item()),
                'mAP50': float(results['map_50'].item()),
                'mAP75': float(results['map_75'].item()),
                'mAP_small': float(results['map_small'].item()),
                'mAP_medium': float(results['map_medium'].item()),
                'mAP_large': float(results['map_large'].item()),
            }
        except Exception as e:
            logger.warning(f"Could not calculate mAP: {e}")
            return {}
    
    def save_checkpoint(self, epoch: int, is_best: bool) -> Path:
        """Save model checkpoint"""
        checkpoint_name = 'best.pt' if is_best else f'epoch_{epoch}.pt'
        checkpoint_path = self.config.output_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'num_classes': self.num_classes
        }
        
        if self.scheduler:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def export_model(self, checkpoint_path: Path) -> Optional[Path]:
        """Export model to ONNX"""
        try:
            logger.info(f"Exporting model to ONNX: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(
                1, 3, self.config.image_size, self.config.image_size
            ).to(self.device)
            
            # Export path
            onnx_path = self.config.output_dir / 'model.onnx'
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['predictions'],
                dynamic_axes={
                    'images': {0: 'batch_size'},
                    'predictions': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Model exported to: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            return None
    
    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)