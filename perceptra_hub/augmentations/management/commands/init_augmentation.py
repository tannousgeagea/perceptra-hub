import random
from django.core.management.base import BaseCommand
from augmentations.models import (
    Augmentation,
    AugmentationParameter,
    AugmentationParameterAssignment
)

class Command(BaseCommand):
    help = "Add common object detection augmentations with their parameters."

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Adding object detection augmentations..."))

        # Define Augmentations & Parameters
        augmentations_data = [
            {
                "name": "horizontal_flip",
                "description": "Flips the image horizontally.",
                "parameters": [],
            },
            {
                "name": "vertical_flip",
                "description": "Flips the image vertically.",
                "parameters": [],
            },
            {
                "name": "rotation",
                "description": "Rotates the image by a specified degree.",
                "parameters": [
                    {"name": "angle", "parameter_type": "integer", "min_value": -30, "max_value": 30, "default_value": 0}
                ],
            },
            {
                "name": "brightness",
                "description": "Adjusts the brightness of the image.",
                "parameters": [
                    {"name": "factor", "parameter_type": "float", "min_value": 0.5, "max_value": 1.5, "default_value": 1.0}
                ],
            },
            {
                "name": "contrast",
                "description": "Adjusts the contrast of the image.",
                "parameters": [
                    {"name": "factor", "parameter_type": "float", "min_value": 0.5, "max_value": 1.5, "default_value": 1.0}
                ],
            },
            {
                "name": "gaussian_noise",
                "description": "Applies Gaussian noise to the image.",
                "parameters": [
                    {"name": "std_dev", "parameter_type": "float", "min_value": 0.0, "max_value": 0.2, "default_value": 0.05}
                ],
            },
            {
                "name": "blur",
                "description": "Blurs the image with a given kernel size.",
                "parameters": [
                    {"name": "kernel_size", "parameter_type": "integer", "min_value": 1, "max_value": 5, "default_value": 3}
                ],
            },
            {
                "name": "scale",
                "description": "Scales the image up or down.",
                "parameters": [
                    {"name": "scale_factor", "parameter_type": "float", "min_value": 0.8, "max_value": 1.2, "default_value": 1.0}
                ],
            },
            {
                "name": "shear",
                "description": "Applies shear transformation to the image.",
                "parameters": [
                    {"name": "shear_angle", "parameter_type": "integer", "min_value": -15, "max_value": 15, "default_value": 0}
                ],
            },
            {
                "name": "cutout",
                "description": "Randomly removes parts of the image.",
                "parameters": [
                    {"name": "num_holes", "parameter_type": "integer", "min_value": 1, "max_value": 10, "default_value": 5},
                    {"name": "hole_size", "parameter_type": "float", "min_value": 0.05, "max_value": 0.2, "default_value": 0.1}
                ],
            }
        ]

        # Create Augmentations & Parameters
        for aug_data in augmentations_data:
            augmentation, created = Augmentation.objects.get_or_create(
                name=aug_data["name"],
                defaults={"description": aug_data["description"], "is_active": True}
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"Added Augmentation: {augmentation.name}"))

            for param_data in aug_data["parameters"]:
                parameter, param_created = AugmentationParameter.objects.get_or_create(
                    name=param_data["name"],
                    defaults={
                        "parameter_type": param_data["parameter_type"],
                        "choices": None  # Used for categorical parameters only
                    }
                )

                if param_created:
                    self.stdout.write(self.style.SUCCESS(f"  Added Parameter: {parameter.name}"))

                # Assign parameters to augmentation
                AugmentationParameterAssignment.objects.get_or_create(
                    augmentation=augmentation,
                    parameter=parameter,
                    defaults={
                        "default_value": param_data["default_value"],
                        "min_value": param_data.get("min_value"),
                        "max_value": param_data.get("max_value")
                    }
                )

        self.stdout.write(self.style.SUCCESS("✅ All augmentations and parameters added successfully!"))
