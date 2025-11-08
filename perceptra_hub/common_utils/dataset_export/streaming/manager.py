
import logging
from django.db import transaction
from django.utils import timezone

from projects.models import Version
from storage.services import get_storage_adapter_for_profile
from .yolo_export import YOLOStreamingExporter
from .base import ExportConfig

logger = logging.getLogger(__name__)

# ============= Export Manager =============

class StreamingDatasetExportManager:
    """Production export manager with streaming."""
    
    EXPORTERS = {
        'yolo': YOLOStreamingExporter,
        # Add more: 'coco': COCOStreamingExporter, etc.
    }
    
    @staticmethod
    def export_version(version_id: int) -> bool:
        """
        Export dataset version with streaming (no local disk).
        
        Thread-safe and atomic.
        """
        version = None
        
        try:
            # Fetch and lock version
            with transaction.atomic():
                version = Version.objects.select_for_update().select_related(
                    'project',
                    'project__organization'
                ).get(id=version_id)
                
                # Prevent concurrent exports
                # if version.export_status == 'processing':
                #     logger.warning(f"Export already in progress for version {version_id}")
                #     return False
                
                version.export_status = 'processing'
                version.save(update_fields=['export_status'])
            
            logger.info(f"Starting export for version {version.version_name}")
            
            # Parse configuration
            config = ExportConfig(**version.export_config)
            
            # Get exporter
            exporter_class = StreamingDatasetExportManager.EXPORTERS.get(
                version.export_format
            )
            
            if not exporter_class:
                raise ValueError(
                    f"Unsupported format: {version.export_format}. "
                    f"Available: {list(StreamingDatasetExportManager.EXPORTERS.keys())}"
                )
            
            # Initialize and run export
            exporter = exporter_class(version, config)
            storage_key = exporter.export()
            
            # Update version with results
            with transaction.atomic():
                version.refresh_from_db()
                version.storage_key = storage_key
                version.export_status = 'completed'
                version.exported_at = timezone.now()
                version.error_log = None
                
                # Update statistics
                version.total_images = exporter.stats.total_images
                version.total_annotations = exporter.stats.total_annotations
                version.train_count = exporter.stats.train_images
                version.val_count = exporter.stats.val_images
                version.test_count = exporter.stats.test_images
                
                # Get file size from storage
                try:
                    storage_profile = version.project.organization.storage_profiles.filter(
                        is_default=True
                    ).first()
                    adapter = get_storage_adapter_for_profile(storage_profile)
                    metadata = adapter.get_file_metadata(storage_key)
                    version.file_size = metadata.size
                except Exception as e:
                    logger.warning(f"Could not get file size: {e}")
                
                version.save()
            
            logger.info(
                f"Export completed successfully. "
                f"Version: {version.version_name}, "
                f"Images: {version.total_images}, "
                f"Size: {version.file_size / (1024*1024):.2f}MB"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"Export failed for version {version_id}: {e}")
            
            # Update version with error
            if version:
                try:
                    with transaction.atomic():
                        version.refresh_from_db()
                        version.export_status = 'failed'
                        version.error_log = str(e)[:1000]  # Limit error message length
                        version.save(update_fields=['export_status', 'error_log'])
                except Exception as save_error:
                    logger.error(f"Failed to save error status: {save_error}")
            
            return False
