from jobs.models import Job, JobImage
from uuid import UUID
from datetime import datetime
from typing import Optional
from projects.models import ProjectImage
from django.db import transaction
from django.db.models import Count
from django.utils.timezone import now

def update_job_status(job: Job):
    images = job.images.select_related("project_image").all()
    statuses = [img.project_image.status for img in images]

    if not job.assignee:
        job.status = "unassigned"
    elif all(s in ["reviewed", "dataset"] for s in statuses):
        job.status = "completed"
    elif all(s in ["annotated", "reviewed"] for s in statuses):
        job.status = "in_review"
    else:
        job.status = "assigned"

    job.save(update_fields=["status"])

def assign_image_to_available_job(project_image: ProjectImage, max_per_job: int = 50) -> Job | None:
    """
    Assigns a single waiting project image to an available job in the same project.
    Updates job image count and project image assignment status.
    Returns the Job it was assigned to or None.
    """
    available_jobs = Job.objects.filter(
        project=project_image.project,
        status__in=["unassigned", "assigned"]
    ).annotate(current_count=Count("images")).order_by("created_at")

    job_image = JobImage.objects.filter(project_image=project_image).first()
    if job_image:
        return job_image.job

    for job in available_jobs:
        if job.image_count is None or job.current_count < max_per_job:
            with transaction.atomic():
                obj, created = JobImage.objects.get_or_create(job=job, project_image=project_image)
                if created:
                    project_image.job_assignment_status = 'assigned'
                    project_image.save(update_fields=["job_assignment_status"])

                    job.image_count = job.current_count + 1
                    job.save(update_fields=['image_count'])

            return job

    latest_job = (
        Job.objects
        .filter(project=project_image.project, name__startswith="Auto Job")
        .order_by("-created_at")
        .first()
    )

    next_number = 1
    if latest_job and latest_job.name.strip().startswith("Auto Job"):
        import re
        match = re.search(r"Auto Job (\d+)", latest_job.name)
        if match:
            next_number = int(match.group(1)) + 1

    new_job_name = f"Auto Job {next_number}"
    with transaction.atomic():
        new_job = Job.objects.create(
            project=project_image.project,
            name=new_job_name,
            description="Automatically created job for new incoming images",
            status="unassigned",
            image_count=1,
        )
        JobImage.objects.create(job=new_job, project_image=project_image)
        project_image.job_assignment_status = 'assigned'
        project_image.save(update_fields=["job_assignment_status"])
        return new_job

def assign_uploaded_image_to_batch(project_image, batch_id: Optional[str], user=None):
    if not batch_id:
        return assign_image_to_available_job(project_image)  # Fallback for production

    latest_job = (
        Job.objects
        .filter(project=project_image.project, name__startswith="Job")
        .order_by("-created_at")
        .first()
    )

    next_number = 1
    if latest_job and latest_job.name.strip().startswith("Job"):
        import re
        match = re.search(r"Job (\d+)", latest_job.name)
        if match:
            next_number = int(match.group(1)) + 1

    new_job_name = f"Job {next_number}"

    job, created = Job.objects.get_or_create(
        project=project_image.project,
        batch_id=batch_id,
        defaults={
            "name": f"{new_job_name}",
            "description": "Batch created via UI image upload",
            "status": "assigned" if user else "unassigned",
            "assignee": user if user else None,
            "image_count": 0
        }
    )

    JobImage.objects.create(job=job, project_image=project_image)
    job.image_count = JobImage.objects.filter(job=job).count()
    job.save(update_fields=["image_count"])

    return job