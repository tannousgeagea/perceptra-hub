
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

from projects.models import VersionImage
from annotations.models import Annotation
from storage.services import get_storage_adapter_for_profile
from .base import BaseExporter

logger = logging.getLogger(__name__)

class YOLOExporter(BaseExporter):
    """YOLO format exporter."""
    
    def export(self) -> Path:
        """Export YOLO dataset."""
        logger.info(f"Starting YOLO export for version {self.version.version_name}")
        
        # Create directory structure
        for split in ['train', 'valid', 'test']:
            
            logger.warning(f"Creating Dir {self.images_dir / split}")
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Get version images grouped by split
        version_images = VersionImage.objects.filter(
            version=self.version
        ).select_related(
            'project_image__image',
            'project_image__image__storage_profile'
        ).prefetch_related(
            'project_image__annotations__annotation_class'
        )
        
        # Collect class names
        class_names = set()
        stats = {'train': 0, 'valid': 0, 'test': 0}
        
        for vi in version_images:
            split = vi.split
            project_image = vi.project_image
            image = project_image.image
            
            # Get annotations
            annotations = Annotation.objects.filter(
                project_image=project_image,
                is_active=True
            ).select_related('annotation_class')
            
            if not self.config.include_predictions:
                annotations = annotations.filter(annotation_source='manual')
            
            if not annotations.exists():
                continue
            
            # Collect class names
            for ann in annotations:
                class_names.add(ann.annotation_class.name)
            
            # Download image
            try:
                adapter = get_storage_adapter_for_profile(image.storage_profile)
                image_data = adapter.download_file(image.storage_key)
            except Exception as e:
                logger.error(f"Failed to download image {image.image_id}: {e}")
                continue
            
            # Convert annotations to YOLO format
            yolo_annotations = []
            for ann in annotations:
                # Data is [xmin, ymin, xmax, ymax] in normalized coords
                xmin, ymin, xmax, ymax = ann.data
                
                # Convert to YOLO format: [x_center, y_center, width, height]
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin
                
                # Filter small annotations
                if width * height < self.config.min_annotation_area:
                    continue
                
                yolo_annotations.append({
                    'class_id': ann.annotation_class.class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
            
            if not yolo_annotations:
                continue
            
            # Process image (with augmentation if enabled)
            apply_aug = split == 'train' and self.config.augment
            processed = self.process_image(image_data, yolo_annotations, apply_aug)
            
            # Save all versions (original + augmented)
            for idx, (img_array, anns) in enumerate(processed):
                suffix = f"_aug{idx}" if idx > 0 else ""
                
                # Save image
                img_filename = f"{image.image_id}{suffix}.jpg"
                img_path = self.images_dir / split / img_filename
                
                img_pil = PILImage.fromarray(img_array)
                img_pil.save(img_path, quality=self.config.image_quality)
                
                # Save labels
                label_filename = f"{image.image_id}{suffix}.txt"
                label_path = self.labels_dir / split / label_filename
                
                with open(label_path, 'w') as f:
                    for ann in anns:
                        f.write(
                            f"{ann['class_id']} "
                            f"{ann['x_center']:.6f} "
                            f"{ann['y_center']:.6f} "
                            f"{ann['width']:.6f} "
                            f"{ann['height']:.6f}\n"
                        )
                
                stats[split] += 1
        
        # Create data.yaml
        class_list = sorted(class_names)
        yaml_content = {
            'path': '../datasets',
            'train': 'images/train',
            'val': 'images/valid',
            'test': 'images/test',
            'names': {i: name for i, name in enumerate(class_list)},
            'nc': len(class_list)
        }
        
        with open(self.temp_dir / 'data.yaml', 'w') as f:
            import yaml
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        # Create README
        self._create_readme(stats, class_list)
        
        # Create zip
        zip_path = self.temp_dir.parent / f"{self.version.version_name}_yolo.zip"
        shutil.make_archive(
            str(zip_path.with_suffix('')),
            'zip',
            self.temp_dir
        )
        
        logger.info(f"YOLO export completed: {stats}")
        return zip_path
    
    def _create_readme(self, stats: Dict, classes: List[str]):
        """Create README file."""
        readme = f"""# {self.version.project.name} - {self.version.version_name}
            ## Dataset Information
            - Format: YOLO
            - Created: {datetime.now().isoformat()}
            - Total Images: {sum(stats.values())}
            - Train: {stats['train']}
            - Validation: {stats['valid']}
            - Test: {stats['test']}

            ## Classes ({len(classes)})
            {chr(10).join(f"{i}: {name}" for i, name in enumerate(classes))}

            ## Directory Structure
            ```
            dataset/
            ├── images/
            │   ├── train/
            │   ├── valid/
            │   └── test/
            ├── labels/
            │   ├── train/
            │   ├── valid/
            │   └── test/
            └── data.yaml
            ```

            ## Usage
            Load with YOLOv8:
            ```python
            from ultralytics import YOLO

            model = YOLO('yolov8n.pt')
            model.train(data='data.yaml', epochs=100)
            ```
        """
        with open(self.temp_dir / 'README.md', 'w') as f:
            f.write(readme)