# Training System Usage Examples

## Overview

The training system is completely **framework-agnostic** and **Django-independent**. You can use it anywhere - CLI scripts, Jupyter notebooks, or integrated into your platform.

---

## Quick Start

### 1. Train YOLO Model

```python
from training.trainers.factory import get_trainer

# Simple API
trainer = get_trainer(
    framework='yolo',
    task='object-detection',
    dataset_path='/data/coco',
    output_dir='/output/yolo_run',
    epochs=100,
    batch_size=16,
    image_size=640,
    # YOLO-specific params
    model_size='n',  # n, s, m, l, x
    version='11',     # 8, 9, 10, 11
    pretrained=True
)

result = trainer.train()
print(f"Best mAP: {result.best_metrics.metrics.get('mAP50')}")
print(f"Checkpoint: {result.best_checkpoint_path}")
print(f"ONNX: {result.onnx_path}")
```

### 2. Train RF-DETR Model

```python
trainer = get_trainer(
    framework='rf-detr',
    task='object-detection',
    dataset_path='/data/coco',
    output_dir='/output/detr_run',
    epochs=50,
    batch_size=8,
    learning_rate=0.0001,
    # RF-DETR specific
    variant='rtdetr-l',  # rtdetr-l, rtdetr-x
    num_classes=80,
    use_amp=True,  # Mixed precision
    scheduler='cosine'
)

result = trainer.train()
```

---

## Advanced Usage

### With Progress Callbacks

```python
from training.trainers.base import TrainingCallbacks
import requests

class WebhookCallbacks(TrainingCallbacks):
    """Send progress updates to webhook"""
    
    def on_epoch_end(self, epoch, metrics):
        requests.post('https://api.myapp.com/training/update', json={
            'epoch': epoch,
            'loss': metrics.train_loss,
            'metrics': metrics.to_dict()
        })

trainer = get_trainer(
    framework='yolo',
    task='object-detection',
    dataset_path='/data',
    output_dir='/output'
)

# Add callbacks
trainer.callbacks = WebhookCallbacks()
result = trainer.train()
```

### From Configuration File

```python
from training.trainers.factory import TrainerFactory
import yaml

# Load config
with open('training_config.yaml') as f:
    config = yaml.safe_load(f)

# Create trainer
trainer = TrainerFactory.from_dict(config)
result = trainer.train()
```

**training_config.yaml:**
```yaml
framework: yolo
task: object-detection
dataset_path: /data/coco
output_dir: /output/run1
epochs: 100
batch_size: 16
learning_rate: 0.001
image_size: 640
device: cuda
workers: 4
optimizer: Adam
scheduler: cosine
augmentation: true

model_params:
  model_size: m
  version: "11"
  pretrained: true
```

---

## Adding New Trainer

Super easy - just inherit from `BaseTrainer` and register:

```python
from training.trainers.base import BaseTrainer
from training.trainers.factory import TrainerRegistry

@TrainerRegistry.register('my-custom-framework')
class MyCustomTrainer(BaseTrainer):
    
    def prepare_dataset(self):
        # Load your dataset
        pass
    
    def create_model(self):
        # Create your model
        pass
    
    def train_epoch(self, epoch):
        # Train one epoch
        return TrainingMetrics(
            epoch=epoch,
            train_loss=0.5,
            metrics={'accuracy': 0.95}
        )
    
    def validate(self, epoch):
        # Validate
        return TrainingMetrics(
            epoch=epoch,
            val_loss=0.4,
            metrics={'val_accuracy': 0.93}
        )
    
    def save_checkpoint(self, epoch, is_best):
        # Save checkpoint
        path = self.config.output_dir / f'checkpoint_{epoch}.pt'
        # ... save logic ...
        return path
    
    def export_model(self, checkpoint_path):
        # Export to ONNX/other format
        return None  # Optional
```

**That's it!** Now you can use it:

```python
trainer = get_trainer(
    framework='my-custom-framework',
    task='my-task',
    dataset_path='/data',
    output_dir='/output'
)
```

---

## Standalone Usage (No Django)

```python
#!/usr/bin/env python
"""
train.py - Standalone training script
"""
from training.trainers.factory import get_trainer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    trainer = get_trainer(
        framework=args.framework,
        task='object-detection',
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs
    )
    
    result = trainer.train()
    
    print(f"✓ Training complete!")
    print(f"  Best checkpoint: {result.best_checkpoint_path}")
    print(f"  Metrics: {result.best_metrics.to_dict()}")

if __name__ == '__main__':
    main()
```

Run:
```bash
python train.py \
  --framework yolo \
  --dataset /data/coco \
  --output /output/run1 \
  --epochs 100
```

---

## Integration with Your Platform

```python
# In your Celery task
from training.trainers.factory import TrainerFactory
from training.trainers.base import TrainingConfig, TrainingCallbacks

# Create config from DB
config = TrainingConfig(
    dataset_path=downloaded_dataset_path,
    output_dir=temp_output_dir,
    **training_session.config  # From DB
)

# Create callbacks for DB updates
class DatabaseCallbacks(TrainingCallbacks):
    def on_epoch_end(self, epoch, metrics):
        # Update TrainingSession in DB
        training_session.current_epoch = epoch
        training_session.current_metrics = metrics.to_dict()
        training_session.save()

# Create and run trainer
trainer = TrainerFactory.create_trainer(
    framework=model.framework.name,
    task=model.task.name,
    config=config,
    callbacks=DatabaseCallbacks()
)

result = trainer.train()

# Upload artifacts
upload_to_storage(result.best_checkpoint_path)
```

---

## Dataset Format Support

### YOLO Format
```
dataset/
  data.yaml
  train/
    images/
    labels/
  val/
    images/
    labels/
```

### COCO Format
```
dataset/
  annotations/
    instances_train.json
    instances_val.json
  train/  (images)
  val/    (images)
```

### Convert Between Formats
```python
from training.trainers.datasets.converters import DatasetConverter

# COCO → YOLO
DatasetConverter.coco_to_yolo(
    coco_ann_file='annotations/instances_train.json',
    output_dir='yolo_labels',
    img_dir='images'
)

# YOLO → COCO
DatasetConverter.yolo_to_coco(
    yolo_dir='labels',
    img_dir='images',
    output_file='coco_annotations.json',
    class_names=['person', 'car', 'dog']
)
```

---

## Benefits

✅ **Framework Agnostic** - Works with any Python project  
✅ **Easy to Extend** - Add new trainers in minutes  
✅ **Consistent API** - Same interface for all frameworks  
✅ **Production Ready** - Proper logging, error handling, checkpointing  
✅ **Flexible** - Use in CLI, notebooks, web apps, anywhere  
✅ **No Vendor Lock-in** - Pure Python, no dependencies on Django  

---

## Available Trainers

| Framework | Status | Tasks | Notes |
|-----------|--------|-------|-------|
| YOLO (v8-11) | ✅ Production | Detection, Segmentation, Classification | Ultralytics |
| RF-DETR | ✅ Production | Detection | RT-DETR implementation |
| Custom | 📦 Template | Any | Extend BaseTrainer |

Add more trainers by extending `BaseTrainer` and registering!