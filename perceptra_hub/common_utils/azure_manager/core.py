import os
import cv2
import numpy as np
import django
django.setup()

from tqdm import tqdm
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from io import BytesIO
import zipfile
from projects.models import Version, VersionImage
from annotations.models import Annotation
from common_utils.data.annotation.core import format_annotation

class AzureManager:
    def __init__(self,):
        """
        Initialize the manager with augmentations.

        Args:
            augmentations (list): List of augmentations to apply.
        """
        pass

    @classmethod
    def download_image_from_azure(self, file_path: str) -> np.ndarray:
        """
        Download an image from Azure Storage.

        Args:
            file_path (str): Path to the image in Azure.

        Returns:
            np.ndarray: Image as a NumPy array.
        """
        with default_storage.open(file_path, 'rb') as file:
            data = file.read()
            np_img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            return np_img
    
    @classmethod
    def upload_image_to_azure(self, local_file_path: str, azure_path: str):
        """
        Upload an image to Azure Storage.

        Args:
            local_file_path (str): Path to the local file.
            azure_path (str): Target path in Azure Storage.
        """
        with open(local_file_path, 'rb') as file:
            content = ContentFile(file.read())
            url = default_storage.save(azure_path, content)
            
        return url
    
    def zip_dataset(self, images, version):
        zip_buffer = BytesIO()
        zip_filename = f"versions/{version.project.name}.v{version.version_number}.zip"
        pbar = tqdm(images, ncols=125)
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for image in pbar:
                zip_prefix = f"{image.project_image.mode.mode}"
                with default_storage.open(image.project_image.image.image_file.name, 'rb') as file:
                    image_bytes = file.read()
                    zipf.writestr(f"{zip_prefix}/images/{image.project_image.image.image_name}.jpg", image_bytes)
                    
                annotations = Annotation.objects.filter(project_image=image.project_image, is_active=True)
                yolo_annotations = [format_annotation(ann, format="yolo") for ann in annotations]
                zipf.writestr(f"{zip_prefix}/labels/{image.project_image.image.image_name}.txt", "".join(yolo_annotations))
        
        zip_buffer.seek(0)
        zip_path = default_storage.save(zip_filename, ContentFile(zip_buffer.getvalue()))
        version.version_file = zip_path
        version.save(update_fields=["version_file"])
        
        
                
    
    