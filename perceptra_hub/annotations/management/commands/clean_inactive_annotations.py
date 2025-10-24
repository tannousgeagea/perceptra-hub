# your_app/management/commands/clean_inactive_annotations.py

from django.core.management.base import BaseCommand
from annotations.models import Annotation

class Command(BaseCommand):
    help = 'Deletes all inactive annotations not associated with any version.'

    def handle(self, *args, **kwargs):
        # Inactive annotations whose project image is not in any version
        inactive_unused_annotations = Annotation.objects.filter(
            is_active=False,
            project_image__associated_versions__isnull=True
        )

        count = inactive_unused_annotations.count()

        if count == 0:
            self.stdout.write(self.style.SUCCESS("No inactive and unused annotations found."))
            return

        inactive_unused_annotations.delete()

        self.stdout.write(self.style.SUCCESS(f"Deleted {count} inactive and unused annotations."))
