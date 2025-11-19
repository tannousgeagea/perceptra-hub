import json
import logging
from typing import Dict, List
from datetime import datetime

import numpy as np

from projects.models import VersionImage
from annotations.models import Annotation
from .base import BaseStreamingExporter

logger = logging.getLogger(__name__)


class COCOStreamingExporter(BaseStreamingExporter):
    """COCO format streaming exporter with augmentation."""
    
    def export(self) -> str:
        """Export COCO dataset directly to storage."""
        logger.info(f"Starting COCO export: {self.version.version_name}")
        
        # Initialize COCO structures for each split
        coco_data = {
            'train': self._init_coco_structure(),
            'val': self._init_coco_structure(),
            'test': self._init_coco_structure()
        }
        
        class_map = {}  # annotation_class.class_id -> coco_category_id
        image_id_counter = {'train': 1, 'val': 1, 'test': 1}
        annotation_id_counter = {'train': 1, 'val': 1, 'test': 1}
        
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
            
            # Add to COCO structure
            self._add_image_and_annotations(
                coco_data[split],
                img_array,
                annotations,
                image,
                split,
                image_id_counter,
                annotation_id_counter,
                class_map
            )
            
            # Generate augmented versions (only for training)
            if split == 'train' and self.config.augment:
                aug_count = self.config.augmentation_factor - 1
                
                for aug_idx, (aug_img, aug_anns) in enumerate(
                    self.generate_augmented_versions(
                        img_array,
                        self._convert_to_normalized(annotations, image.width, image.height),
                        aug_count
                    )
                ):
                    # Convert back to absolute coordinates
                    abs_anns = self._denormalize_annotations(
                        aug_anns,
                        aug_img.shape[1],
                        aug_img.shape[0]
                    )
                    
                    self._add_augmented_image(
                        coco_data[split],
                        aug_img,
                        abs_anns,
                        f"{image.image_id}_aug{aug_idx + 1}",
                        split,
                        image_id_counter,
                        annotation_id_counter,
                        annotations  # Original annotations for class info
                    )
                    
                    self.stats.augmented_images += 1
                    self.stats.train_annotations += len(abs_anns)
        
        # Finalize categories
        for split_data in coco_data.values():
            split_data['categories'] = [
                {
                    'id': coco_id,
                    'name': name,
                    'supercategory': 'object'
                }
                for name, coco_id in class_map.items()
            ]
        
        # Save JSON files and images
        for split, data in coco_data.items():
            if data['images']:
                # Save annotations JSON
                json_content = json.dumps(data, indent=2).encode('utf-8')
                self.add_file(f"annotations/instances_{split}.json", json_content)
        
        # Add README
        self._add_readme(class_map)
        
        # Stream to storage
        storage_key = f"org-{self.version.project.organization.slug}/projects/{self.version.project.project_id}/datasets/{self.version.version_name}_coco.zip"
        file_size = self.stream_to_storage(storage_key)
        
        logger.info(
            f"COCO export completed. "
            f"Total: {self.stats.total_images} images, "
            f"Augmented: {self.stats.augmented_images}, "
            f"Size: {file_size / (1024*1024):.2f}MB"
        )
        
        return storage_key
    
    def _init_coco_structure(self) -> Dict:
        """Initialize COCO JSON structure."""
        return {
            'info': {
                'description': f'{self.version.project.name} - {self.version.version_name}',
                'version': '1.0',
                'year': datetime.now().year,
                'date_created': datetime.now().isoformat()
            },
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': []
        }
    
    def _add_image_and_annotations(
        self,
        coco_data: Dict,
        img_array: np.ndarray,
        annotations: List[Annotation],
        image,
        split: str,
        image_id_counter: Dict,
        annotation_id_counter: Dict,
        class_map: Dict
    ):
        """Add image and annotations to COCO structure."""
        img_id = image_id_counter[split]
        height, width = img_array.shape[:2]
        
        # Add image entry
        coco_data['images'].append({
            'id': img_id,
            'file_name': f"{image.image_id}.jpg",
            'width': width,
            'height': height
        })
        
        # Save image file
        img_bytes = self.image_to_bytes(img_array)
        self.add_file(f"{split}/{image.image_id}.jpg", img_bytes)
        
        # Add annotations
        for ann in annotations:
            # Register class
            class_name = ann.annotation_class.name
            if class_name not in class_map:
                class_map[class_name] = len(class_map) + 1
            
            xmin, ymin, xmax, ymax = ann.data
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            
            # Filter small annotations
            if bbox_width * bbox_height < self.config.min_annotation_area:
                continue
            
            coco_data['annotations'].append({
                'id': annotation_id_counter[split],
                'image_id': img_id,
                'category_id': class_map[class_name],
                'bbox': [xmin, ymin, bbox_width, bbox_height],
                'area': bbox_width * bbox_height,
                'iscrowd': 0
            })
            
            annotation_id_counter[split] += 1
        
        # Update stats
        self._update_stats(split, len(annotations))
        image_id_counter[split] += 1
    
    def _add_augmented_image(
        self,
        coco_data: Dict,
        img_array: np.ndarray,
        annotations: List[Dict],
        filename: str,
        split: str,
        image_id_counter: Dict,
        annotation_id_counter: Dict,
        original_annotations: List[Annotation]
    ):
        """Add augmented image to COCO structure."""
        img_id = image_id_counter[split]
        height, width = img_array.shape[:2]
        
        coco_data['images'].append({
            'id': img_id,
            'file_name': f"{filename}.jpg",
            'width': width,
            'height': height
        })
        
        img_bytes = self.image_to_bytes(img_array)
        self.add_file(f"{split}/{filename}.jpg", img_bytes)
        
        # Map class_ids back to class names
        class_id_to_ann = {ann.annotation_class.class_id: ann for ann in original_annotations}
        
        for ann in annotations:
            orig_ann = class_id_to_ann.get(ann['class_id'])
            if not orig_ann:
                continue
            
            coco_data['annotations'].append({
                'id': annotation_id_counter[split],
                'image_id': img_id,
                'category_id': ann['class_id'],
                'bbox': [
                    ann['xmin'],
                    ann['ymin'],
                    ann['width'],
                    ann['height']
                ],
                'area': ann['width'] * ann['height'],
                'iscrowd': 0
            })
            
            annotation_id_counter[split] += 1
        
        image_id_counter[split] += 1
    
    def _convert_to_normalized(
        self,
        annotations: List[Annotation],
        img_width: int,
        img_height: int
    ) -> List[Dict]:
        """Convert annotations to normalized format for augmentation."""
        normalized = []
        for ann in annotations:
            xmin, ymin, xmax, ymax = ann.data
            normalized.append({
                'class_id': ann.annotation_class.class_id,
                'x_center': (xmin + xmax) / 2 / img_width,
                'y_center': (ymin + ymax) / 2 / img_height,
                'width': (xmax - xmin) / img_width,
                'height': (ymax - ymin) / img_height
            })
        return normalized
    
    def _denormalize_annotations(
        self,
        annotations: List[Dict],
        img_width: int,
        img_height: int
    ) -> List[Dict]:
        """Convert normalized annotations back to absolute coordinates."""
        denormalized = []
        for ann in annotations:
            x_center = ann['x_center'] * img_width
            y_center = ann['y_center'] * img_height
            width = ann['width'] * img_width
            height = ann['height'] * img_height
            
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            
            denormalized.append({
                'class_id': ann['class_id'],
                'xmin': xmin,
                'ymin': ymin,
                'width': width,
                'height': height
            })
        return denormalized
    
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
    
    def _add_readme(self, class_map: Dict):
        """Add README to zip."""
        augmentation_info = ""
        if self.config.augment:
            augmentation_info = f"""
                ## Augmentation
                - Enabled: Yes
                - Augmentation Factor: {self.config.augmentation_factor}x
                - Total Augmented Images: {self.stats.augmented_images}
                """
                        
            readme = f"""# {self.version.project.name} - {self.version.version_name}

                ## Dataset Information
                - Format: COCO
                - Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                - Total Images: {self.stats.total_images}
                - Total Annotations: {self.stats.total_annotations}

                ## Split Distribution
                - Train: {self.stats.train_images} images ({self.stats.train_annotations} annotations)
                - Validation: {self.stats.val_images} images ({self.stats.val_annotations} annotations)
                - Test: {self.stats.test_images} images ({self.stats.test_annotations} annotations)
                {augmentation_info}

                ## Classes ({len(class_map)})
                ```
                {chr(10).join(f"{coco_id}: {name}" for name, coco_id in sorted(class_map.items(), key=lambda x: x[1]))}
                ```

                ## Directory Structure
                ```
                dataset/
                ├── train/
                │   └── *.jpg
                ├── val/
                │   └── *.jpg
                ├── test/
                │   └── *.jpg
                ├── annotations/
                │   ├── instances_train.json
                │   ├── instances_val.json
                │   └── instances_test.json
                └── README.md
                ```

                ## Usage with Detectron2
                ```python
                from detectron2.data.datasets import register_coco_instances

                register_coco_instances("my_dataset_train", {{}}, 
                                    "annotations/instances_train.json", "train")
                register_coco_instances("my_dataset_val", {{}}, 
                                    "annotations/instances_val.json", "val")
                ```
            """
        self.add_file('README.md', readme.encode('utf-8'))