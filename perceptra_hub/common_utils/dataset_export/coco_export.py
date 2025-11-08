
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
import numpy as np

from projects.models import VersionImage
from annotations.models import Annotation
from storage.services import get_storage_adapter_for_profile
from .base import BaseExporter

logger = logging.getLogger(__name__)


class COCOExporter(BaseExporter):
    """COCO format exporter."""
    
    def export(self) -> Path:
        """Export COCO dataset."""
        logger.info(f"Starting COCO export for version {self.version.version_name}")
        
        # Create directory structure
        self.images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir = self.temp_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize COCO structure for each split
        coco_data = {
            'train': self._init_coco_structure(),
            'val': self._init_coco_structure(),
            'test': self._init_coco_structure()
        }
        
        image_id = 1
        annotation_id = 1
        category_map = {}
        
        # Get version images
        version_images = VersionImage.objects.filter(
            version=self.version
        ).select_related(
            'project_image__image',
            'project_image__image__storage_profile'
        ).prefetch_related(
            'project_image__annotations__annotation_class'
        )
        
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
            
            # Download image
            try:
                adapter = get_storage_adapter_for_profile(image.storage_profile)
                image_data = adapter.download_file(image.storage_key)
            except Exception as e:
                logger.error(f"Failed to download image {image.image_id}: {e}")
                continue
            
            # Save image
            img_filename = f"{image.image_id}.jpg"
            img_path = self.images_dir / img_filename
            
            with open(img_path, 'wb') as f:
                f.write(image_data)
            
            # Add image to COCO
            coco_data[split]['images'].append({
                'id': image_id,
                'file_name': img_filename,
                'width': image.width,
                'height': image.height,
                'date_captured': image.created_at.isoformat()
            })
            
            # Add annotations
            for ann in annotations:
                # Add category if not exists
                class_name = ann.annotation_class.name
                if class_name not in category_map:
                    cat_id = len(category_map) + 1
                    category_map[class_name] = cat_id
                    
                    for split_data in coco_data.values():
                        split_data['categories'].append({
                            'id': cat_id,
                            'name': class_name,
                            'supercategory': 'object'
                        })
                
                # Convert to COCO bbox format: [x, y, width, height] in pixels
                xmin, ymin, xmax, ymax = ann.data
                x = xmin * image.width
                y = ymin * image.height
                width = (xmax - xmin) * image.width
                height = (ymax - ymin) * image.height
                area = width * height
                
                # Filter small annotations
                if area < self.config.min_annotation_area * image.width * image.height:
                    continue
                
                coco_data[split]['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_map[class_name],
                    'bbox': [x, y, width, height],
                    'area': area,
                    'iscrowd': 0,
                    'segmentation': []
                })
                
                annotation_id += 1
            
            image_id += 1
        
        # Save COCO JSON files
        for split, data in coco_data.items():
            with open(annotations_dir / f"instances_{split}.json", 'w') as f:
                json.dump(data, f, indent=2)
        
        # Create README
        stats = {
            split: len(data['images'])
            for split, data in coco_data.items()
        }
        self._create_readme(stats, list(category_map.keys()))
        
        # Create zip
        zip_path = self.temp_dir.parent / f"{self.version.version_name}_coco.zip"
        shutil.make_archive(
            str(zip_path.with_suffix('')),
            'zip',
            self.temp_dir
        )
        
        logger.info(f"COCO export completed: {stats}")
        return zip_path
    
    def _init_coco_structure(self) -> Dict:
        """Initialize COCO JSON structure."""
        return {
            'info': {
                'description': f'{self.version.project.name} Dataset',
                'version': self.version.version_name,
                'year': datetime.now().year,
                'date_created': datetime.now().isoformat()
            },
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': []
        }
    
    def _create_readme(self, stats: Dict, classes: List[str]):
        """Create README file."""
        readme = f"""# {self.version.project.name} - {self.version.version_name}
            ## Dataset Information
            - Format: COCO
            - Created: {datetime.now().isoformat()}
            - Total Images: {sum(stats.values())}
            - Train: {stats['train']}
            - Validation: {stats['val']}
            - Test: {stats['test']}

            ## Classes ({len(classes)})
            {chr(10).join(f"- {name}" for name in classes)}

            ## Directory Structure
            ```
            dataset/
            ├── images/
            └── annotations/
                ├── instances_train.json
                ├── instances_val.json
                └── instances_test.json
            ```

            ## Usage
            Load with pycocotools:
            ```python
            from pycocotools.coco import COCO

            coco = COCO('annotations/instances_train.json')
            ```
        """
        with open(self.temp_dir / 'README.md', 'w') as f:
            f.write(readme)