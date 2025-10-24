# yourapp/management/commands/compress_augmented_images.py
import io
import os
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from augmentations.models import VersionImageAugmentation  # adjust import as needed
from PIL import Image as PilImage

class Command(BaseCommand):
    help = "Compress all augmented images and replace them with a compressed version."

    def handle(self, *args, **options):
        augmentations = VersionImageAugmentation.objects.all()
        total = augmentations.count()
        self.stdout.write(f"Found {total} augmented images to compress.")

        for aug_obj in augmentations:
            if not aug_obj.augmented_image_file:
                self.stdout.write(f"Skipping {aug_obj} (no augmented image file).")
                continue

            try:
                old_path = aug_obj.augmented_image_file.name
                self.stdout.write(f"Processing: {old_path}")

                with default_storage.open(old_path, 'rb') as f:
                    pil_img = PilImage.open(f)
                    pil_img.load()

                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG", quality=60, optimize=True)
                buffer.seek(0)

                compressed_file = ContentFile(buffer.read())
                new_filename = os.path.basename(old_path)

                if default_storage.exists(old_path):
                    default_storage.delete(old_path)

                aug_obj.augmented_image_file.save(new_filename, compressed_file, save=False)
                aug_obj.save()

                self.stdout.write(self.style.SUCCESS(f"Compressed augmentation: {aug_obj}"))
            except Exception as e:
                self.stderr.write(f"Error compressing augmentation {aug_obj}: {e}")
