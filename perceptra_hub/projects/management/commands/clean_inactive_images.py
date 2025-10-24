# your_app/management/commands/clean_inactive_images.py

from django.core.management.base import BaseCommand
from projects.models import ProjectImage

class Command(BaseCommand):
    help = 'Deletes all inactive project images not used in any version.'

    def handle(self, *args, **kwargs):
        # Filter images that are inactive and not used in any version
        unused_images = ProjectImage.objects.filter(
            is_active=False,
            associated_versions__isnull=True
        )

        count = unused_images.count()

        if count == 0:
            self.stdout.write(self.style.SUCCESS("No inactive and unused project images found."))
            return

        unused_images.delete()

        self.stdout.write(self.style.SUCCESS(f"Deleted {count} inactive and unused project images."))
