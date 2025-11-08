
import json
import logging
import yaml
from typing import Dict, List
from datetime import datetime

import numpy as np

from projects.models import VersionImage
from annotations.models import Annotation
from .base import BaseStreamingExporter

logger = logging.getLogger(__name__)

# ============= YOLO Exporter =============

class YOLOStreamingExporter(BaseStreamingExporter):
    """YOLO format streaming exporter with augmentation."""
    
    def export(self) -> str:
        """Export YOLO dataset directly to storage."""
        logger.info(f"Starting YOLO export: {self.version.version_name}")
        
        class_names = set()
        
        # Get all version images
        version_images = VersionImage.objects.filter(
            version=self.version
        ).select_related(
            'project_image__image',
            'project_image__image__storage_profile'
        ).prefetch_related(
            'project_image__annotations__annotation_class'
        )
        
        total_images = version_images.count()
        
        for idx, vi in enumerate(version_images):
            if idx % 10 == 0:
                logger.info(f"Processing {idx}/{total_images} images")
            
            split = self.normalize_split(vi.split)
            project_image = vi.project_image
            image = project_image.image
            
            # Get annotations
            annotations = self.get_annotations(project_image)
            
            if not annotations:
                self.stats.skipped_images += 1
                continue
            
            # Download image
            image_data = self.download_image(image)
            if not image_data:
                continue
            
            # Process image
            img_array = self.process_image_bytes(
                image_data,
                target_size=self.config.image_size
            )
            
            # Convert annotations to YOLO format
            yolo_annotations = self._convert_to_yolo(
                annotations,
                image.width,
                image.height
            )
            
            if not yolo_annotations:
                self.stats.skipped_images += 1
                continue
            
            # Collect class names
            for ann in annotations:
                class_names.add(ann.annotation_class.name)
            
            # Save original image and labels
            self._save_image_and_labels(
                img_array,
                yolo_annotations,
                f"{image.image_id}",
                split
            )
            
            # Update stats
            self._update_stats(split, len(yolo_annotations))
            
            # Generate augmented versions (only for training)
            if split == 'train' and self.config.augment:
                aug_count = self.config.augmentation_factor - 1
                
                for aug_idx, (aug_img, aug_anns) in enumerate(
                    self.generate_augmented_versions(
                        img_array,
                        yolo_annotations,
                        aug_count
                    )
                ):
                    self._save_image_and_labels(
                        aug_img,
                        aug_anns,
                        f"{image.image_id}_aug{aug_idx + 1}",
                        split
                    )
                    
                    self.stats.augmented_images += 1
                    self.stats.train_annotations += len(aug_anns)
        
        # Add metadata files
        self._add_metadata_files(sorted(class_names))
        
        # Stream to storage
        storage_key = f"org-{self.version.project.organization.slug}/projects/{self.version.project.project_id}/datasets/{self.version.version_name}_yolo.zip"
        file_size = self.stream_to_storage(storage_key)
        
        logger.info(
            f"YOLO export completed. "
            f"Total: {self.stats.total_images} images, "
            f"Augmented: {self.stats.augmented_images}, "
            f"Size: {file_size / (1024*1024):.2f}MB"
        )
        
        return storage_key
    
    def _convert_to_yolo(
        self,
        annotations: List[Annotation],
        img_width: int,
        img_height: int
    ) -> List[Dict]:
        """Convert annotations to YOLO format."""
        yolo_annotations = []
        
        for ann in annotations:
            xmin, ymin, xmax, ymax = ann.data
            
            # Convert to YOLO format (normalized coordinates)
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
        
        return yolo_annotations
    
    def _save_image_and_labels(
        self,
        img_array: np.ndarray,
        annotations: List[Dict],
        filename: str,
        split: str
    ):
        """Save image and labels to zip."""
        # Save image
        img_bytes = self.image_to_bytes(img_array)
        self.add_file(f"{split}/images/{filename}.jpg", img_bytes)
        
        # Save labels
        label_lines = []
        for ann in annotations:
            label_lines.append(
                f"{ann['class_id']} "
                f"{ann['x_center']:.6f} {ann['y_center']:.6f} "
                f"{ann['width']:.6f} {ann['height']:.6f}\n"
            )
        
        label_content = ''.join(label_lines).encode('utf-8')
        self.add_file(f"{split}/labels/{filename}.txt", label_content)
    
    def _update_stats(self, split: str, ann_count: int):
        """Update export statistics."""
        if split == 'train':
            self.stats.train_images += 1
            self.stats.train_annotations += ann_count
        elif split == 'val':
            self.stats.val_images += 1
            self.stats.val_annotations += ann_count
        elif split == 'test':
            self.stats.test_images += 1
            self.stats.test_annotations += ann_count
    
    def _add_metadata_files(self, class_names: List[str]):
        """Add data.yaml and README to zip."""
        # Create data.yaml
        yaml_content = {
            'path': '../datasets',
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {i: name for i, name in enumerate(class_names)},
            'nc': len(class_names)
        }
        
        yaml_bytes = yaml.dump(
            yaml_content,
            default_flow_style=False,
            allow_unicode=True
        ).encode('utf-8')
        self.add_file('data.yaml', yaml_bytes)
        
        # Create README
        readme = self._create_readme(class_names)
        self.add_file('README.md', readme.encode('utf-8'))
    
    def _create_readme(self, classes: List[str]) -> str:
        """Create README content."""
        augmentation_info = ""
        if self.config.augment:
            augmentation_info = f"""
            ## Augmentation
            - Enabled: Yes
            - Augmentation Factor: {self.config.augmentation_factor}x
            - Total Augmented Images: {self.stats.augmented_images}
            - Configuration: {json.dumps(self.config.augmentation_config, indent=2)}
            """
        
        return f"""# {self.version.project.name} - {self.version.version_name}

            ## Dataset Information
            - Format: YOLO
            - Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - Total Images: {self.stats.total_images}
            - Total Annotations: {self.stats.total_annotations}

            ## Split Distribution
            - Train: {self.stats.train_images} images ({self.stats.train_annotations} annotations)
            - Validation: {self.stats.val_images} images ({self.stats.val_annotations} annotations)
            - Test: {self.stats.test_images} images ({self.stats.test_annotations} annotations)
            {augmentation_info}

            ## Classes ({len(classes)})
            ```
            {chr(10).join(f"{i}: {name}" for i, name in enumerate(classes))}
            ```

            ## Configuration
            - Image Size: {self.config.image_size or 'Original'}
            - Image Quality: {self.config.image_quality}%
            - Min Annotation Area: {self.config.min_annotation_area}

            ## Directory Structure
            ```
            dataset/
            ├── train/
            │   ├── images/
            │   └── labels/
            ├── val/
            │   ├── images/
            │   └── labels/
            ├── test/
            │   ├── images/
            │   └── labels/
            ├── data.yaml
            └── README.md
            ```

            ## Usage with YOLOv8
            ```python
            from ultralytics import YOLO

            # Train
            model = YOLO('yolov8n.pt')
            results = model.train(data='data.yaml', epochs=100, imgsz=640)

            # Validate
            metrics = model.val()

            # Predict
            results = model.predict('path/to/image.jpg')
            ```
        """