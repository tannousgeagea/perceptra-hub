"""
Dataset utilities for training.
Location: training/trainers/datasets/
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
from typing import Tuple, Dict, Any, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============= COCO Dataset =============
# Location: training/trainers/datasets/coco_dataset.py

class COCODetectionDataset(Dataset):
    """
    COCO format dataset for object detection.
    Works with standard COCO JSON annotations.
    """
    
    def __init__(
        self,
        img_dir: Path,
        ann_file: Path,
        transforms=None
    ):
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        
        # Load COCO annotations
        with open(ann_file) as f:
            coco_data = json.load(f)
        
        self.images = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: cat for cat in coco_data['categories']}
        
        # Group annotations by image
        self.img_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_annotations:
                self.img_annotations[img_id] = []
            self.img_annotations[img_id].append(ann)
        
        self.img_ids = list(self.images.keys())
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.img_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations
        annotations = self.img_annotations.get(img_id, [])
        
        # Convert to boxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64)
        }
        
        # Apply transforms
        if self.transforms:
            # For albumentations
            if hasattr(self.transforms, '__call__'):
                import numpy as np
                image = np.array(image)
                
                transformed = self.transforms(
                    image=image,
                    bboxes=boxes,
                    labels=labels
                )
                
                image = transformed['image']
                target['boxes'] = torch.as_tensor(
                    transformed['bboxes'],
                    dtype=torch.float32
                )
        
        return image, target


# ============= Transforms =============
# Location: training/trainers/datasets/transforms.py

def get_transforms(
    image_size: int = 640,
    is_train: bool = True,
    augment: bool = True
):
    """
    Get image transforms for training/validation.
    Uses albumentations for efficient augmentation.
    
    Args:
        image_size: Target image size
        is_train: Whether for training (enables augmentation)
        augment: Whether to apply augmentation
    
    Returns:
        Albumentations transform pipeline
    """
    if is_train and augment:
        # Training transforms with augmentation
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                value=(114, 114, 114)
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.CLAHE(p=0.1),
            A.HueSaturationValue(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))
    else:
        # Validation transforms (no augmentation)
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                value=(114, 114, 114)
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))


# ============= Dataset Converters =============
# Location: training/trainers/datasets/converters.py

class DatasetConverter:
    """
    Convert between different dataset formats.
    Useful for supporting multiple frameworks.
    """
    
    @staticmethod
    def coco_to_yolo(
        coco_ann_file: Path,
        output_dir: Path,
        img_dir: Path
    ):
        """
        Convert COCO format to YOLO format.
        
        YOLO format:
          - One .txt file per image
          - Each line: class_id center_x center_y width height (normalized)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(coco_ann_file) as f:
            coco_data = json.load(f)
        
        images = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        img_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
        
        # Convert each image's annotations
        for img_id, img_info in images.items():
            img_width = img_info['width']
            img_height = img_info['height']
            
            annotations = img_annotations.get(img_id, [])
            
            # Write YOLO format file
            txt_filename = Path(img_info['file_name']).stem + '.txt'
            txt_path = output_dir / txt_filename
            
            with open(txt_path, 'w') as f:
                for ann in annotations:
                    # Convert COCO bbox to YOLO format
                    x, y, w, h = ann['bbox']
                    class_id = ann['category_id'] - 1  # YOLO uses 0-indexed
                    
                    # Normalize and convert to center format
                    center_x = (x + w / 2) / img_width
                    center_y = (y + h / 2) / img_height
                    norm_w = w / img_width
                    norm_h = h / img_height
                    
                    f.write(
                        f"{class_id} {center_x:.6f} {center_y:.6f} "
                        f"{norm_w:.6f} {norm_h:.6f}\n"
                    )
        
        print(f"Converted {len(images)} images to YOLO format at {output_dir}")
    
    @staticmethod
    def yolo_to_coco(
        yolo_dir: Path,
        img_dir: Path,
        output_file: Path,
        class_names: list[str]
    ):
        """
        Convert YOLO format to COCO format.
        """
        yolo_dir = Path(yolo_dir)
        img_dir = Path(img_dir)
        
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for idx, name in enumerate(class_names):
            coco_data['categories'].append({
                'id': idx + 1,
                'name': name,
                'supercategory': 'object'
            })
        
        ann_id = 1
        
        # Process each annotation file
        for txt_file in yolo_dir.glob('*.txt'):
            img_filename = txt_file.stem + '.jpg'  # Assume jpg
            img_path = img_dir / img_filename
            
            if not img_path.exists():
                continue
            
            # Get image dimensions
            img = Image.open(img_path)
            img_width, img_height = img.size
            img_id = len(coco_data['images']) + 1
            
            # Add image info
            coco_data['images'].append({
                'id': img_id,
                'file_name': img_filename,
                'width': img_width,
                'height': img_height
            })
            
            # Read YOLO annotations
            with open(txt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0]) + 1  # COCO uses 1-indexed
                    center_x = float(parts[1]) * img_width
                    center_y = float(parts[2]) * img_height
                    box_w = float(parts[3]) * img_width
                    box_h = float(parts[4]) * img_height
                    
                    # Convert to COCO format [x, y, width, height]
                    x = center_x - box_w / 2
                    y = center_y - box_h / 2
                    
                    coco_data['annotations'].append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': class_id,
                        'bbox': [x, y, box_w, box_h],
                        'area': box_w * box_h,
                        'iscrowd': 0
                    })
                    
                    ann_id += 1
        
        # Save COCO JSON
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Converted {len(coco_data['images'])} images to COCO format: {output_file}")