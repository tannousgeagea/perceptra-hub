import cv2
import json
import hashlib
import numpy as np
import albumentations as A
from albumentations import CoarseDropout
from pathlib import Path
from typing import List, Dict, Union
from .utils import (
    save_annotations, save_image
)

class AugmentationPipeline:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def hash_image(self, image):
        """
        Generate a hash for the image to detect duplicates.

        Args:
            image (np.ndarray): Image array.

        Returns:
            str: MD5 hash of the image.
        """
        return hashlib.md5(image.tobytes()).hexdigest()


    def create_preprocessing_pipeline(self, preprocess_settings: List[Dict]) -> A.Compose:
        """
        Create a preprocessing pipeline.

        Args:
            preprocess_settings (List[Dict]): Preprocessing settings.

        Returns:
            A.Compose: Preprocessing pipeline.
        """
        transforms = []
        for setting in preprocess_settings:
            if setting["name"] == "resize":
                transforms.append(A.Resize(**setting["params"]))
        return A.Compose(transforms)

    def create_augmentation_pipeline(self, augmentations: List[Dict]) -> A.Compose:
        """
        Create an augmentation pipeline.

        Args:
            augmentations (List[Dict]): Augmentation settings.

        Returns:
            A.Compose: Augmentation pipeline.
        """
        transforms = []
        for aug in augmentations:
            if aug["name"] == "horizontal_flip":
                transforms.append(A.HorizontalFlip(**aug["params"]))
            elif aug["name"] == "vertical_flip":
                transforms.append(A.VerticalFlip(**aug["params"]))
            elif aug["name"] == "rotate":
                transforms.append(A.Affine(**aug["params"]))
            elif aug["name"] == "gaussian_blur":
                transforms.append(A.GaussianBlur(**aug["params"]))
            elif aug["name"] == "brightness":
                transforms.append(A.RandomBrightnessContrast(**aug["params"]))
            elif aug["name"] == "contrast":
                transforms.append(A.RandomBrightnessContrast(**aug["params"]))
            elif aug["name"] == "random_crop":
                transforms.append(A.RandomCrop(**aug["params"]))
            elif aug["name"] == "scale":
                transforms.append(A.RandomScale(**aug["params"]))
            elif aug["name"] == "shear":
                transforms.append(A.Affine(**aug["params"]))
            elif aug["name"] == "sharpen":
                transforms.append(A.Sharpen(**aug["params"]))
            elif aug["name"] == "color_jitter":
                transforms.append(A.ColorJitter(**aug["params"]))
            elif aug["name"] == "elastic_transform":
                transforms.append(A.ElasticTransform(**aug["params"]))
            elif aug["name"] == "grid_distortion":
                transforms.append(A.GridDistortion(**aug["params"]))
            elif aug["name"] == "optical_distortion":
                transforms.append(A.OpticalDistortion(**aug["params"]))
            elif aug["name"] == "motion_blur":
                transforms.append(A.MotionBlur(**aug["params"]))
            elif aug["name"] == "random_shadow":
                transforms.append(A.RandomShadow(**aug["params"]))
            elif aug["name"] == "random_sun_flare":
                transforms.append(A.RandomSunFlare(**aug["params"]))
            elif aug["name"] == "hue_saturation_value":
                transforms.append(A.HueSaturationValue(**aug["params"]))
            elif aug["name"] == "cutout":
                transforms.append(CoarseDropout(**aug["params"]))
            else:
                raise ValueError(f"Unsupported augmentation: {aug['name']}")

        return A.Compose(
            transforms, bbox_params=A.BboxParams(format="albumentations", label_fields=["category_ids"])
        )

    def apply_augmentations_policy(
        self,
        image,
        annotations,
        annotation_type,
        file_name_prefix,
        user_selected_augmentations,
        multiplier=1,
        preprocess_pipeline=None,
    ):
        """
        Apply augmentations based on the augmentation policy.

        Args:
            image (np.ndarray): Original image.
            annotations (dict): Original annotations.
            annotation_type (str): Type of annotations ('detection', 'segmentation', 'classification').
            file_name_prefix (str): Prefix for augmented file naming.
            user_selected_augmentations (list): List of augmentations and parameters.
            multiplier (int): Number of augmented versions to generate for each source image.
            preprocess_pipeline (A.Compose): Preprocessing pipeline to apply before augmentations.

        Returns:
            List[str]: List of augmented image paths.
        """
        augmented_images = []
        seen_hashes = set()

        if preprocess_pipeline:
            preprocessed = preprocess_pipeline(image=image)
            image_preprocessed = preprocessed["image"]
            hash_val = self.hash_image(image_preprocessed)
            if hash_val not in seen_hashes:
                seen_hashes.add(hash_val)
                path = save_image(self.output_dir, f"{file_name_prefix}_preprocessed", image_preprocessed)
                augmented_images.append(path)

        augmentation_pipeline = self.create_augmentation_pipeline(user_selected_augmentations)
        for i in range(multiplier - 1):
            augmented = augmentation_pipeline(
                image=image,
                bboxes=annotations.get("bboxes", []),
                category_ids=annotations.get("labels", []),
            )

            augmented_image = augmented["image"]
            hash_val = self.hash_image(augmented_image)

            # Avoid duplicates
            if hash_val not in seen_hashes:
                seen_hashes.add(hash_val)
                path = save_image(self.output_dir, f"{file_name_prefix}_augmented_{i + 1}", augmented_image)
                annotation_path = save_annotations(
                    self.output_dir, f"{file_name_prefix}_augmented_{i + 1}", augmented["bboxes"], augmented["category_ids"], annotation_type=annotation_type
                )
                augmented_images.append({
                    "image": path,
                    "label": annotation_path
                })

        return augmented_images


PREDEFINED_AUGMENTATIONS = [
    {"name": "horizontal_flip", "params": {"p": 0.5}},
    {"name": "vertical_flip", "params": {"p": 0.5}},
    {"name": "rotate", "params": {"rotate": 15, "p": 0.5}},
    {"name": "gaussian_blur", "params": {"blur_limit": 3, "p": 0.5}},
    {"name": "hue_saturation_value", "params": {"hue_shift_limit": 25, "p": 0.5}},
    {"name": "shear", "params": {"shear": (-10, 10), "p": 0.5}},
    {"name": "cutout", "params": {"num_holes_range": (1, 3), "hole_height_range": (50, 200), "hole_width_range": (50, 200), "fill": 0, "p": 1.0}}
]


# Example usage
if __name__ == "__main__":
    import numpy as np
    import django
    django.setup()
    
    from common_utils.azure_manager.core import AzureAugmentationManager
    from projects.models import (
        ProjectImage
    )
    
    from annotations.models import Annotation
    azure_manager = AzureAugmentationManager()
    pipeline = AugmentationPipeline(output_dir="./augmented_dataset")

    # Dummy image and annotations
    image = ProjectImage.objects.filter(project__name="amk_front_impurity").last()
    
    cv_image = azure_manager.download_image_from_azure(
        file_path=image.image.image_file.name
    )
    
    print(cv_image.shape)
    annotations = Annotation.objects.filter(project_image = image)
    annotations = {
        "bboxes": [
            ann.data for ann in annotations
        ],
        "labels": [
            ann.annotation_class.class_id for ann in annotations
        ] 
    }
    # User settings
    print(annotations)
    
    preprocess_settings = [{"name": "resize", "params": {"height": 640, "width": 640}}]
    augmentations = [
        {"name": "horizontal_flip", "params": {"p": 0.5}},
        {"name": "vertical_flip", "params": {"p": 0.5}},
        {"name": "rotate", "params": {"rotate": 45, "p": 0.5}},
        {"name": "gaussian_blur", "params": {"blur_limit": 3, "p": 0.5}},
        {"name": "hue_saturation_value", "params": {"hue_shift_limit": 20, "p": 0.5}},
        {"name": "shear", "params": {"shear": (-15, 15), "p": 1}},
        {"name": "cutout", "params": {"num_holes_range": (1, 3), "hole_height_range": (100, 200), "hole_width_range": (100, 200), "fill": 0, "p": 1.0}}
    ]

    preprocess_pipeline = pipeline.create_preprocessing_pipeline(preprocess_settings)

    # Apply policy
    pipeline.apply_augmentations_policy(
        image=cv_image,
        annotations=annotations,
        annotation_type="detection",
        file_name_prefix="example_image",
        user_selected_augmentations=augmentations,
        multiplier=3,
        preprocess_pipeline=preprocess_pipeline,
    )
