import os
from django.conf import settings
from django.core.management.base import BaseCommand
from storages.backends.azure_storage import AzureStorage
from projects.models import ProjectImage  # Update with your actual app and model

class Command(BaseCommand):
    help = 'Transfer local images to Azure Blob Storage'

    def handle(self, *args, **kwargs):
        azure_storage = AzureStorage()
        local_storage_path = settings.MEDIA_ROOT  # Assuming media files are stored here

        # Get all project images
        # image_id = "6619b498-481a-4233-97bd-743a6c3d9d1a"
        images = ProjectImage.objects.filter(project__name='amk_front_impurity')

        for image in images:
            # Extract the relative file path
            relative_path = image.image.image_file.name  # Assuming `file` is the field storing image
            local_path = os.path.join(local_storage_path, relative_path)

            if os.path.exists(local_path):
                # Open the file and upload it to Azure
                with open(local_path, 'rb') as file_data:
                    azure_storage.save(relative_path, file_data)
                self.stdout.write(self.style.SUCCESS(f'Uploaded: {relative_path}'))
            else:
                self.stdout.write(self.style.ERROR(f'File not found: {local_path}'))