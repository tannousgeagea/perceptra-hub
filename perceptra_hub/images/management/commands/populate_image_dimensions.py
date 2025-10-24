import os
from django.core.management.base import BaseCommand
from PIL import Image as PILImage
from io import BytesIO
import requests
from images.models import Image  # Adjust import as needed
from django.core.files.storage import default_storage

class Command(BaseCommand):
    help = "Populate width and height for images stored in Azure Blob Storage"

    def handle(self, *args, **options):
        updated = 0
        skipped = 0
        errored = 0

        qs = Image.objects.filter(width__isnull=True, height__isnull=True)

        self.stdout.write(f"Processing {qs.count()} images...")

        for img in qs:
            try:
                file_path = img.image_file.name  # relative path in storage
                if not default_storage.exists(file_path):
                    self.stderr.write(f"⚠️ File not found in storage: {file_path}")
                    skipped += 1
                    continue

                with default_storage.open(file_path, 'rb') as f:
                    with PILImage.open(f) as pil_img:
                        img.width, img.height = pil_img.size
                        img.save(update_fields=['width', 'height'])
                        self.stdout.write(self.style.SUCCESS(f"Update image size: {pil_img.size}"))


                        updated += 1

            except Exception as e:
                self.stderr.write(f"❌ Error processing Image ID {img.id} ({img.image_file.name}): {e}")
                errored += 1

        self.stdout.write(self.style.SUCCESS(
            f"✅ Done: {updated} updated, {skipped} skipped, {errored} failed"
        ))
