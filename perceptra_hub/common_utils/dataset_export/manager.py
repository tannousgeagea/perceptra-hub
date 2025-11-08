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
from django.utils import timezone
from django.core.files.base import ContentFile

from projects.models import Version
from .base import ExportConfig
from .yolo_export import YOLOExporter
from .coco_export import COCOExporter
from .pascalvoc_export import PascalVOCExporter
logger = logging.getLogger(__name__)

# ============= Export Manager =============

class DatasetExportManager:
    """Manages dataset export process."""
    
    EXPORTERS = {
        'yolo': YOLOExporter,
        'coco': COCOExporter,
        'pascal_voc': PascalVOCExporter,
    }
    
    @staticmethod
    def export_version(version_id: int) -> bool:
        """
        Export dataset version.
        
        Args:
            version_id: Version primary key
            
        Returns:
            bool: Success status
        """
        try:
            version = Version.objects.select_related('project').get(id=version_id)
            
            # Update status to processing
            version.export_status = 'processing'
            version.save(update_fields=['export_status'])
            
            logger.info(f"Starting export for version {version.version_name}")
            
            # Parse export config
            config = ExportConfig(**version.export_config)
            
            # Create temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Get exporter
                exporter_class = DatasetExportManager.EXPORTERS.get(version.export_format)
                if not exporter_class:
                    raise ValueError(f"Unsupported format: {version.export_format}")
                
                exporter = exporter_class(version, config, temp_path)
                
                # Export dataset
                zip_path = exporter.export()
                
                # Upload to storage or save locally
                with open(zip_path, 'rb') as f:
                    version.dataset_file.save(
                        zip_path.name,
                        ContentFile(f.read()),
                        save=False
                    )
                
                # Update version
                version.file_size = zip_path.stat().st_size
                version.export_status = 'completed'
                version.exported_at = timezone.now()
                version.error_log = None
                version.save()
            
            logger.info(f"Export completed for version {version.version_name}")
            return True
            
        except Exception as e:
            logger.exception(f"Export failed for version {version_id}: {e}")
            
            # Update version with error
            try:
                version = Version.objects.get(id=version_id)
                version.export_status = 'failed'
                version.error_log = str(e)
                version.save(update_fields=['export_status', 'error_log'])
            except:
                pass
            
            return False
