import logging
from typing import Dict, List
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

from projects.models import VersionImage
from annotations.models import Annotation
from .base import BaseStreamingExporter

logger = logging.getLogger(__name__)


class VOCStreamingExporter(BaseStreamingExporter):
    """Pascal VOC format streaming exporter with augmentation."""
    
    def export(self) -> str:
        """Export Pascal VOC dataset directly to storage."""
        logger.info(f"Starting Pascal VOC export: {self.version.version_name}")
        
        class_names = set()
        split_files = {'train': [], 'val': [], 'test': []}
        
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
            
            # Collect class names
            for ann in annotations:
                class_names.add(ann.annotation_class.name)
            
            # Save image and annotation
            filename = str(image.image_id)
            self._save_image_and_xml(
                img_array,
                annotations,
                filename,
                split
            )
            
            split_files[split].append(filename)
            self._update_stats(split, len(annotations))
            
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
                        aug_img.shape[0],
                        annotations  # Original annotations for class info
                    )
                    
                    aug_filename = f"{image.image_id}_aug{aug_idx + 1}"
                    self._save_augmented_image_and_xml(
                        aug_img,
                        abs_anns,
                        aug_filename,
                        split
                    )
                    
                    split_files[split].append(aug_filename)
                    self.stats.augmented_images += 1
                    self.stats.train_annotations += len(abs_anns)
        
        # Add ImageSets
        self._add_imagesets(split_files)
        
        # Add class labels
        self._add_class_labels(sorted(class_names))
        
        # Add README
        self._add_readme(sorted(class_names))
        
        # Stream to storage
        storage_key = f"org-{self.version.project.organization.slug}/projects/{self.version.project.project_id}/datasets/{self.version.version_name}_voc.zip"
        file_size = self.stream_to_storage(storage_key)
        
        logger.info(
            f"Pascal VOC export completed. "
            f"Total: {self.stats.total_images} images, "
            f"Augmented: {self.stats.augmented_images}, "
            f"Size: {file_size / (1024*1024):.2f}MB"
        )
        
        return storage_key
    
    def _save_image_and_xml(
        self,
        img_array: np.ndarray,
        annotations: List[Annotation],
        filename: str,
        split: str
    ):
        """Save image and XML annotation."""
        height, width = img_array.shape[:2]
        
        # Save image
        img_bytes = self.image_to_bytes(img_array)
        self.add_file(f"JPEGImages/{filename}.jpg", img_bytes)
        
        # Create XML
        xml_content = self._create_voc_xml(
            filename,
            width,
            height,
            annotations
        )
        self.add_file(f"Annotations/{filename}.xml", xml_content.encode('utf-8'))
    
    def _save_augmented_image_and_xml(
        self,
        img_array: np.ndarray,
        annotations: List[Dict],
        filename: str,
        split: str
    ):
        """Save augmented image and XML annotation."""
        height, width = img_array.shape[:2]
        
        # Save image
        img_bytes = self.image_to_bytes(img_array)
        self.add_file(f"JPEGImages/{filename}.jpg", img_bytes)
        
        # Create XML
        xml_content = self._create_voc_xml_from_dict(
            filename,
            width,
            height,
            annotations
        )
        self.add_file(f"Annotations/{filename}.xml", xml_content.encode('utf-8'))
    
    def _create_voc_xml(
        self,
        filename: str,
        width: int,
        height: int,
        annotations: List[Annotation]
    ) -> str:
        """Create Pascal VOC XML annotation."""
        root = ET.Element('annotation')
        
        ET.SubElement(root, 'folder').text = 'JPEGImages'
        ET.SubElement(root, 'filename').text = f'{filename}.jpg'
        
        source = ET.SubElement(root, 'source')
        ET.SubElement(source, 'database').text = self.version.project.name
        
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = '3'
        
        ET.SubElement(root, 'segmented').text = '0'
        
        for ann in annotations:
            xmin, ymin, xmax, ymax = ann.data
            
            # Filter small annotations
            bbox_area = (xmax - xmin) * (ymax - ymin)
            if bbox_area < self.config.min_annotation_area:
                continue
            
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = ann.annotation_class.name
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(xmin))
            ET.SubElement(bndbox, 'ymin').text = str(int(ymin))
            ET.SubElement(bndbox, 'xmax').text = str(int(xmax))
            ET.SubElement(bndbox, 'ymax').text = str(int(ymax))
        
        return self._prettify_xml(root)
    
    def _create_voc_xml_from_dict(
        self,
        filename: str,
        width: int,
        height: int,
        annotations: List[Dict]
    ) -> str:
        """Create Pascal VOC XML from dict annotations."""
        root = ET.Element('annotation')
        
        ET.SubElement(root, 'folder').text = 'JPEGImages'
        ET.SubElement(root, 'filename').text = f'{filename}.jpg'
        
        source = ET.SubElement(root, 'source')
        ET.SubElement(source, 'database').text = self.version.project.name
        
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = '3'
        
        ET.SubElement(root, 'segmented').text = '0'
        
        for ann in annotations:
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = ann['class_name']
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(ann['xmin'] * width))
            ET.SubElement(bndbox, 'ymin').text = str(int(ann['ymin'] * height))
            ET.SubElement(bndbox, 'xmax').text = str(int(ann['xmax'] * width))
            ET.SubElement(bndbox, 'ymax').text = str(int(ann['ymax'] * height))
        
        return self._prettify_xml(root)
    
    def _prettify_xml(self, elem: ET.Element) -> str:
        """Return a pretty-printed XML string."""
        rough_string = ET.tostring(elem, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ", encoding='utf-8').decode('utf-8')
    
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
                'class_name': ann.annotation_class.name,
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
        img_height: int,
        original_annotations: List[Annotation]
    ) -> List[Dict]:
        """Convert normalized annotations back to absolute coordinates."""
        # Create class_id to class_name mapping
        class_map = {ann.annotation_class.class_id: ann.annotation_class.name 
                     for ann in original_annotations}
        
        denormalized = []
        for ann in annotations:
            x_center = ann['x_center'] * img_width
            y_center = ann['y_center'] * img_height
            width = ann['width'] * img_width
            height = ann['height'] * img_height
            
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = xmin + width
            ymax = ymin + height
            
            denormalized.append({
                'class_id': ann['class_id'],
                'class_name': class_map.get(ann['class_id'], 'unknown'),
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
        return denormalized
    
    def _add_imagesets(self, split_files: Dict[str, List[str]]):
        """Add ImageSets split files."""
        for split, filenames in split_files.items():
            if filenames:
                content = '\n'.join(filenames) + '\n'
                self.add_file(
                    f"ImageSets/Main/{split}.txt",
                    content.encode('utf-8')
                )
    
    def _add_class_labels(self, class_names: List[str]):
        """Add class labels file."""
        content = '\n'.join(class_names) + '\n'
        self.add_file('labels.txt', content.encode('utf-8'))
    
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
    
    def _add_readme(self, class_names: List[str]):
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
            - Format: Pascal VOC
            - Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - Total Images: {self.stats.total_images}
            - Total Annotations: {self.stats.total_annotations}

            ## Split Distribution
            - Train: {self.stats.train_images} images ({self.stats.train_annotations} annotations)
            - Validation: {self.stats.val_images} images ({self.stats.val_annotations} annotations)
            - Test: {self.stats.test_images} images ({self.stats.test_annotations} annotations)
            {augmentation_info}

            ## Classes ({len(class_names)})
            ```
            {chr(10).join(class_names)}
            ```

            ## Directory Structure
            ```
            dataset/
            ├── JPEGImages/
            │   └── *.jpg
            ├── Annotations/
            │   └── *.xml
            ├── ImageSets/
            │   └── Main/
            │       ├── train.txt
            │       ├── val.txt
            │       └── test.txt
            ├── labels.txt
            └── README.md
            ```

            ## Usage with PyTorch
            ```python
            from torch.utils.data import Dataset
            import xml.etree.ElementTree as ET

            class VOCDataset(Dataset):
                def __init__(self, root, image_set='train'):
                    self.root = root
                    with open(f'{{root}}/ImageSets/Main/{{image_set}}.txt') as f:
                        self.ids = [line.strip() for line in f]
                
                def __getitem__(self, idx):
                    img_id = self.ids[idx]
                    img = Image.open(f'{{self.root}}/JPEGImages/{{img_id}}.jpg')
                    xml_path = f'{{self.root}}/Annotations/{{img_id}}.xml'
                    # Parse XML for annotations
                    return img, annotations
            ```
        """
        self.add_file('README.md', readme.encode('utf-8'))