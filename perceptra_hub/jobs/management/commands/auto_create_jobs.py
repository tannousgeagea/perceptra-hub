from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.timezone import now

from projects.models import Project, ProjectImage
from jobs.models import Job, JobImage


class Command(BaseCommand):
    help = "Auto-create jobs and assign project images by their current status."

    def handle(self, *args, **options):
        self.stdout.write("🚀 Auto-assigning images to jobs by status...")

        status_to_jobstatus = {
            'unannotated': "unassigned",
            'annotated': "assigned",
            'reviewed': "in_review",
            'dataset': "completed",
        }

        with transaction.atomic():
            for project in Project.objects.all():
                
                i = 1
                for image_status, job_status in status_to_jobstatus.items():
                    # Only select images that are not yet linked to any job
                    project_images = ProjectImage.objects.filter(
                        project=project,
                        status=image_status,
                    ).exclude(job_assignments__isnull=False)

                    if not project_images.exists():
                        continue

                    job = Job.objects.create(
                        project=project,
                        name=f"Job {i}",
                        description=f"Auto-created job for {image_status} images.",
                        status=job_status,
                        image_count=project_images.count(),
                        created_at=now(),
                        updated_at=now(),
                    )

                    job_images = [
                        JobImage(job=job, project_image=img)
                        for img in project_images
                    ]
                    JobImage.objects.bulk_create(job_images)
                    i = i + 1

                    self.stdout.write(
                        self.style.SUCCESS(
                            f"✅ Created job '{job.name}' with {project_images.count()} images for project '{project.name}'."
                        )
                    )

        self.stdout.write(self.style.SUCCESS("🎉 Done! All eligible images have been assigned to jobs."))
