import os
import cv2
import shutil
import zipfile
import django
django.setup()
import logging
import numpy as np
from io import BytesIO
import concurrent.futures
from pathlib import Path
from celery import shared_task
from django.core.cache import cache
from django.core.files.base import ContentFile
from projects.models import Version, VersionImage
from annotations.models import Annotation
from django.core.files.storage import default_storage
from common_utils.data.annotation.core import format_annotation
from common_utils.augmentation.core import AugmentationPipeline

PREDEFINED_AUGMENTATIONS = [
    {"name": "horizontal_flip", "params": {"p": 0.5}},
    {"name": "vertical_flip", "params": {"p": 0.5}},
    {"name": "rotate", "params": {"rotate": 45, "p": 0.5}},
    {"name": "gaussian_blur", "params": {"blur_limit": 3, "p": 0.5}},
    {"name": "hue_saturation_value", "params": {"hue_shift_limit": 20, "p": 0.5}},
    {"name": "shear", "params": {"shear": (-15, 15), "p": 1}},
    {"name": "cutout", "params": {"num_holes_range": (1, 3), "hole_height_range": (100, 200), "hole_width_range": (100, 200), "fill": 0, "p": 1.0}}
]

def decode_image(file_bytes):
    """Convert file bytes into a cv2 image (BGR)"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@shared_task(bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5}, ignore_result=True,
             name='create_version:execute')
def execute(self, version_id, image_ids, **kwargs):
    try:
        version = Version.objects.get(id=version_id)
        total = len(image_ids)
        augmented_output_dir = Path("/tmp/augmented_dataset")
        augmented_output_dir.mkdir(parents=True, exist_ok=True)
        aug_pipeline = AugmentationPipeline(output_dir=str(augmented_output_dir))
        augmented_files = []
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            def process_image(image_id):
                version_image = VersionImage.objects.get(id=image_id)
                prefix = version_image.project_image.mode.mode
                image_field = version_image.project_image.image.image_file.name
                with default_storage.open(image_field, 'rb') as file:
                    image_bytes = file.read()

                image_name = os.path.basename(version_image.project_image.image.image_name)
                annotations = Annotation.objects.filter(
                    project_image=version_image.project_image, is_active=True
                )
                yolo_annotations = "".join([format_annotation(ann, format="yolo") for ann in annotations])

                augmented_files = []
                if PREDEFINED_AUGMENTATIONS and not version_image.project_image.marked_as_null and version_image.project_image.mode.mode == "train":
                    cv_image = decode_image(image_bytes)
                    ann_dict = {
                        "bboxes": [ann.data for ann in annotations],
                        "labels": [ann.annotation_class.class_id for ann in annotations]
                    }
                    augmented_files = aug_pipeline.apply_augmentations_policy(
                        image=cv_image,
                        annotations=ann_dict,
                        annotation_type="detection",
                        file_name_prefix=image_name,
                        user_selected_augmentations=PREDEFINED_AUGMENTATIONS,
                        multiplier=3,
                    )

                return (prefix, image_name, image_bytes, yolo_annotations, augmented_files)

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_image = {executor.submit(process_image, img_id): img_id for img_id in image_ids}
                processed = 0
                for future in concurrent.futures.as_completed(future_to_image):
                    try:
                        prefix, image_name, image_bytes, yolo_annotations, augmented_files = future.result()
                        zipf.writestr(f"{prefix}/images/{image_name}.jpg", image_bytes)
                        zipf.writestr(f"{prefix}/labels/{image_name}.txt", yolo_annotations)

                        for aug_file in augmented_files:
                            image_path = Path(aug_file['image'])
                            label_path = Path(aug_file['label'])
                            if not image_path.exists():
                                continue

                            with open(image_path, "rb") as aug_f:
                                aug_bytes = aug_f.read()

                            zipf.writestr(f"{prefix}/images/{image_path.name}", aug_bytes)
                            if label_path.exists():
                                with open(label_path, "rb") as ann_f:
                                    aug_ann_bytes = ann_f.read()
                                zipf.writestr(f"{prefix}/labels/{label_path.name}", aug_ann_bytes)

                    except Exception as e:
                        logging.error(f"Failed to zip file: {e}")
                    
                    processed += 1
                    cache.set(f"task_progress_{self.request.id}", {'current': processed, 'total': total}, timeout=3600)
        
        zip_buffer.seek(0)
        zip_filename = f"{version.project.name}.v{version.version_number}.zip"
        # zip_path = default_storage.save(zip_filename, ContentFile(zip_buffer.getvalue()))
        # version.version_file = zip_path
        # version.save(update_fields=["version_file"])

        local_versions_dir = "/tmp/versions"
        os.makedirs(local_versions_dir, exist_ok=True)
        local_path = os.path.join(local_versions_dir, zip_filename)
        with open(local_path, "wb") as f:
            f.write(zip_buffer.getvalue())

        shutil.rmtree(str(augmented_output_dir))
        cache.set(f"task_progress_{self.request.id}", {'current': total, 'total': total, 'status': 'Completed'}, timeout=3600)
        return {'current': total, 'total': total, 'status': 'Completed'}
    
    except Exception as err:
        raise ValueError(f"Error saving delivery data into db: {err}")