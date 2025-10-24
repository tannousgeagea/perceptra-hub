# myapp/management/commands/compress_images.py
import io
import os
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from images.models import Image 
from PIL import Image as PilImage

class Command(BaseCommand):
    help = "Compress all existing images and replace them with a compressed version."

    def handle(self, *args, **options):
        images = Image.objects.all()
        total = images.count()
        self.stdout.write(f"Found {total} images to compress.")
        
        for img_obj in images:
            try:
                with default_storage.open(img_obj.image_file.name, 'rb') as f:
                    pil_img = PilImage.open(f)
                    pil_img.load()

                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG", quality=60, optimize=True)
                buffer.seek(0)

                compressed_file = ContentFile(buffer.read())
                filename = img_obj.image_file.name

                if default_storage.exists(filename):
                    default_storage.delete(filename)

                img_obj.image_file.save(os.path.basename(filename), compressed_file, save=False)
                img_obj.save()

                self.stdout.write(self.style.SUCCESS(f"Compressed image {img_obj.image_name}"))
            except Exception as e:
                self.stderr.write(f"Error compressing image {img_obj.image_name}: {e}")
