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
from .base import BaseExporter
logger = logging.getLogger(__name__)

class PascalVOCExporter(BaseExporter):
    """Pascal VOC format exporter."""
    
    def export(self) -> Path:
        """Export Pascal VOC dataset."""
        logger.info(f"Starting Pascal VOC export for version {self.version.version_name}")
        
        # Create directory structure
        (self.temp_dir / "JPEGImages").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "Annotations").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)
        
        train_list = []
        val_list = []
        test_list = []
        
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
            img_path = self.temp_dir / "JPEGImages" / img_filename
            
            with open(img_path, 'wb') as f:
                f.write(image_data)
            
            # Create XML annotation
            self._create_voc_xml(image, annotations)
            
            # Add to split list
            if split == 'train':
                train_list.append(str(image.image_id))
            elif split == 'val':
                val_list.append(str(image.image_id))
            else:
                test_list.append(str(image.image_id))
        
        # Save split lists
        with open(self.temp_dir / "ImageSets" / "Main" / "train.txt", 'w') as f:
            f.write('\n'.join(train_list))
        
        with open(self.temp_dir / "ImageSets" / "Main" / "val.txt", 'w') as f:
            f.write('\n'.join(val_list))
        
        with open(self.temp_dir / "ImageSets" / "Main" / "test.txt", 'w') as f:
            f.write('\n'.join(test_list))
        
        # Create README
        stats = {
            'train': len(train_list),
            'val': len(val_list),
            'test': len(test_list)
        }
        self._create_readme(stats)
        
        # Create zip
        zip_path = self.temp_dir.parent / f"{self.version.version_name}_voc.zip"
        shutil.make_archive(
            str(zip_path.with_suffix('')),
            'zip',
            self.temp_dir
        )
        
        logger.info(f"Pascal VOC export completed: {stats}")
        return zip_path
    
    def _create_voc_xml(self, image: Image, annotations):
        """Create Pascal VOC XML annotation file."""
        root = ET.Element('annotation')
        
        ET.SubElement(root, 'folder').text = 'JPEGImages'
        ET.SubElement(root, 'filename').text = f"{image.image_id}.jpg"
        
        source = ET.SubElement(root, 'source')
        ET.SubElement(source, 'database').text = self.version.project.name
        
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(image.width)
        ET.SubElement(size, 'height').text = str(image.height)
        ET.SubElement(size, 'depth').text = '3'
        
        ET.SubElement(root, 'segmented').text = '0'
        
        for ann in annotations:
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = ann.annotation_class.name
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            
            # Convert normalized coords to pixels
            xmin, ymin, xmax, ymax = ann.data
            xmin_px = int(xmin * image.width)
            ymin_px = int(ymin * image.height)
            xmax_px = int(xmax * image.width)
            ymax_px = int(ymax * image.height)
            
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(xmin_px)
            ET.SubElement(bndbox, 'ymin').text = str(ymin_px)
            ET.SubElement(bndbox, 'xmax').text = str(xmax_px)
            ET.SubElement(bndbox, 'ymax').text = str(ymax_px)
        
        # Save XML
        tree = ET.ElementTree(root)
        xml_path = self.temp_dir / "Annotations" / f"{image.image_id}.xml"
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    
    def _create_readme(self, stats: Dict):
        """Create README file."""
        readme = f"""# {self.version.project.name} - {self.version.version_name}

            ## Dataset Information
            - Format: Pascal VOC
            - Created: {datetime.now().isoformat()}
            - Total Images: {sum(stats.values())}
            - Train: {stats['train']}
            - Validation: {stats['val']}
            - Test: {stats['test']}

            ## Directory Structure
            ```
            dataset/
            ├── JPEGImages/
            ├── Annotations/
            └── ImageSets/
                └── Main/
                    ├── train.txt
                    ├── val.txt
                    └── test.txt
            ```
        """
        with open(self.temp_dir / 'README.md', 'w') as f:
            f.write(readme)