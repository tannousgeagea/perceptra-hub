# ml_models/management/commands/seed_tags.py
from django.core.management.base import BaseCommand
from ml_models.models import ModelTag

DEFAULT_TAGS = [
    {"name": "classification", "description": "Used for classification tasks"},
    {"name": "object-detection", "description": "Used for object detection tasks"},
    {"name": "segmentation", "description": "Used for segmentation tasks"},
    {"name": "production", "description": "Deployed model in production"},
    {"name": "baseline", "description": "Reference or benchmark model"},
    {"name": "experimental", "description": "Experimental or test model"},
]

class Command(BaseCommand):
    help = "Seeds default model tags into one organization (tags are per organization)"

    def add_arguments(self, parser):
        parser.add_argument("--org", required=True, help="Organization slug or org_id")

    def handle(self, *args, **options):
        import uuid
        from django.core.management.base import CommandError
        from organizations.models import Organization

        value = options["org"]
        try:
            lookup = {"org_id": uuid.UUID(value)}
        except ValueError:
            lookup = {"slug": value}
        organization = Organization.objects.filter(**lookup).first()
        if organization is None:
            raise CommandError(f"No organization with slug or id '{value}'")

        for tag in DEFAULT_TAGS:
            obj, created = ModelTag.objects.get_or_create(
                organization=organization, name=tag["name"], defaults={"description": tag["description"]}
            )
            action = "Created" if created else "Exists"
            self.stdout.write(self.style.SUCCESS(f"{action} tag: {tag['name']}"))
