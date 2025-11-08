"""
Production-ready dataset export system with multiple formats and augmentation.
"""
import os
import json
import shutil
import zipfile
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET
from io import BytesIO
from PIL import Image as PILImage
import albumentations as A
import cv2
import numpy as np

from django.db import transaction
from django.utils import timezone
from django.core.files.base import ContentFile

from projects.models import Version, VersionImage, ProjectImage
from annotations.models import Annotation
from images.models import Image
from storage.services import get_storage_adapter_for_profile

logger = logging.getLogger(__name__)


# ============= Export Configuration =============

@dataclass
class ExportConfig:
    """Export configuration."""
    image_size: Optional[int] = None  # Resize to square (e.g., 640)
    image_quality: int = 95
    normalize: bool = False  # Normalize coordinates to 0-1
    include_difficult: bool = True
    include_predictions: bool = False  # Include model predictions
    min_annotation_area: float = 0.0  # Filter small annotations
    
    # Augmentation
    augment: bool = False
    augmentation_factor: int = 1  # Number of augmented copies per image
    augmentation_config: Optional[Dict] = None


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_limit: int = 15
    brightness_contrast: bool = True
    blur: bool = False
    noise: bool = False
    random_crop: bool = False


# ============= Format Exporters =============

class BaseExporter:
    """Base exporter class."""
    
    def __init__(self, version: Version, config: ExportConfig, temp_dir: Path):
        self.version = version
        self.config = config
        self.temp_dir = temp_dir
        self.images_dir = temp_dir / "images"
        self.labels_dir = temp_dir / "labels"
        
    def export(self) -> Path:
        """Export dataset and return zip path."""
        raise NotImplementedError
        
    def get_augmentation_pipeline(self) -> A.Compose:
        """Create augmentation pipeline."""
        transforms = []
        
        aug_config = self.config.augmentation_config or {}
        
        if aug_config.get('horizontal_flip', True):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if aug_config.get('vertical_flip', False):
            transforms.append(A.VerticalFlip(p=0.5))
        
        if aug_config.get('rotation_limit', 0) > 0:
            transforms.append(A.Rotate(limit=aug_config['rotation_limit'], p=0.5))
        
        if aug_config.get('brightness_contrast', True):
            transforms.append(A.RandomBrightnessContrast(p=0.5))
        
        if aug_config.get('blur', False):
            transforms.append(A.Blur(blur_limit=3, p=0.3))
        
        if aug_config.get('noise', False):
            transforms.append(A.GaussNoise(p=0.3))
        
        # Bounding box parameters for albumentation
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            )
        )
    
    def process_image(
        self,
        image_data: bytes,
        annotations: List[Dict],
        apply_augmentation: bool = False
    ) -> List[tuple]:
        """
        Process image and annotations.
        Returns list of (image_array, annotations) tuples.
        """
        # Load image
        img = PILImage.open(BytesIO(image_data))
        img_array = np.array(img)
        
        # Resize if configured
        if self.config.image_size:
            img_array = cv2.resize(img_array, (self.config.image_size, self.config.image_size))
        
        results = [(img_array, annotations)]
        
        # Apply augmentation if requested
        if apply_augmentation and self.config.augment:
            transform = self.get_augmentation_pipeline()
            
            for _ in range(self.config.augmentation_factor):
                # Convert annotations to albumentation format
                bboxes = []
                class_labels = []
                
                for ann in annotations:
                    # YOLO format: [x_center, y_center, width, height]
                    bboxes.append([
                        ann['x_center'],
                        ann['y_center'],
                        ann['width'],
                        ann['height']
                    ])
                    class_labels.append(ann['class_id'])
                
                # Apply augmentation
                transformed = transform(
                    image=img_array.copy(),
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                # Convert back to annotation format
                aug_annotations = []
                for bbox, class_id in zip(transformed['bboxes'], transformed['class_labels']):
                    aug_annotations.append({
                        'class_id': class_id,
                        'x_center': bbox[0],
                        'y_center': bbox[1],
                        'width': bbox[2],
                        'height': bbox[3]
                    })
                
                results.append((transformed['image'], aug_annotations))
        
        return results


# ============= Celery Task (Optional) =============

# Uncomment if using Celery
# from celery import shared_task
# 
# @shared_task(bind=True, max_retries=3)
# def export_dataset_task(self, version_id: int):
#     """Celery task for dataset export."""
#     try:
#         return DatasetExportManager.export_version(version_id)
#     except Exception as exc:
#         self.retry(exc=exc, countdown=60)