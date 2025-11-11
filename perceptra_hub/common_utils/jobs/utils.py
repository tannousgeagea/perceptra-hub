from jobs.models import Job, JobImage
from uuid import UUID
from datetime import datetime
from typing import Optional
from projects.models import ProjectImage
from django.db import transaction
from django.db.models import Count
from django.core.cache import cache
from django.utils.timezone import now
import logging
import re

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


def _get_next_job_number(project, prefix: str) -> int:
    """
    Helper to get the next sequential job number for a given prefix.
    Returns 1 if no jobs with this prefix exist.
    """
    latest_job = (
        Job.objects
        .filter(project=project, name__startswith=prefix)
        .order_by("-created_at")
        .first()
    )
    
    if not latest_job:
        return 1
    
    match = re.search(rf"{re.escape(prefix)} (\d+)", latest_job.name)
    return int(match.group(1)) + 1 if match else 1

def _assign_to_job(project_image, job, user=None) -> None:
    """
    Helper to create JobImage relationship and update statuses.
    Should be called within a transaction.
    """
    job_image, created = JobImage.objects.get_or_create(
        job=job,
        project_image=project_image,
        defaults={
            "created_by": user,
            "updated_by": user,
        }
    )
    
    if not created:
        return  # Already assigned to this job
    
    # Update project image status
    project_image.job_assignment_status = 'assigned'
    project_image.save(update_fields=["job_assignment_status"])
    
    # Update job image count
    job.image_count = JobImage.objects.filter(job=job).count()
    job.updated_by = user
    job.save(update_fields=['image_count', 'updated_by'])

def _get_or_create_batch_job(project, batch_id: str, user=None) -> 'Job':
    """
    Thread-safe get_or_create for batch jobs using external lock.
    Uses cache-based locking to prevent race conditions across multiple workers.
    """
    lock_key = f"batch_job_lock:{project.id}:{batch_id}"
    lock_timeout = 30  # seconds
    
    # Try to acquire lock with timeout
    lock_acquired = cache.add(lock_key, "locked", lock_timeout)
    
    try:
        if not lock_acquired:
            # Another process is creating the job, wait and retry
            import time
            max_retries = 10
            retry_delay = 0.5
            
            for _ in range(max_retries):
                time.sleep(retry_delay)
                existing_job = Job.objects.filter(
                    project=project,
                    batch_id=batch_id
                ).first()
                
                if existing_job:
                    logging.info(f"Found batch job after waiting: {existing_job.name}")
                    return existing_job
            
            # Timeout waiting for other process
            raise TimeoutError(f"Timeout waiting for batch job creation: {batch_id}")
        
        # We have the lock, check if job exists
        existing_job = Job.objects.filter(
            project=project,
            batch_id=batch_id
        ).first()
        
        if existing_job:
            logging.info(f"Found existing batch job: {existing_job.name}")
            return existing_job
        
        # Create new job
        next_number = _get_next_job_number(project, "Job")
        
        logging.info(f"Creating new batch job for batch_id {batch_id}")
        
        with transaction.atomic():
            new_job = Job.objects.create(
                project=project,
                batch_id=batch_id,
                name=f"Job {next_number}",
                description="Batch created via UI image upload",
                status="assigned" if user else "unassigned",
                assignee=user if user else None,
                image_count=0,
                created_by=user,
                updated_by=user,
            )
        
        return new_job
        
    finally:
        # Always release the lock
        if lock_acquired:
            cache.delete(lock_key)

def assign_image_to_available_job(
    project_image,
    max_per_job: int = 50,
    user=None,
) -> Optional['Job']:
    """
    Assigns a project image to an available auto-job.
    
    Logic:
    1. Check if image is already assigned
    2. Find available jobs with space
    3. If none, create a new auto job
    
    Returns the Job it was assigned to or None.
    """
    # Check if already assigned
    existing_assignment = JobImage.objects.filter(
        project_image=project_image
    ).select_related('job').first()
    
    if existing_assignment:
        return existing_assignment.job
    
    # Find available jobs with space
    available_jobs = (
        Job.objects
        .filter(
            project=project_image.project,
            status__in=["unassigned", "assigned"]
        )
        .annotate(current_count=Count("images"))
        .filter(current_count__lt=max_per_job)
        .order_by("created_at")
    )
    
    # Try to assign to existing job
    for job in available_jobs:
        with transaction.atomic():
            _assign_to_job(project_image, job, user)
        return job
    
    # Create new auto job if no space
    next_number = _get_next_job_number(project_image.project, "Auto Job")
    
    with transaction.atomic():
        new_job = Job.objects.create(
            project=project_image.project,
            name=f"Auto Job {next_number}",
            description="Automatically created job for incoming images",
            status="unassigned",
            image_count=0,
            created_by=user,
            updated_by=user,
        )
        _assign_to_job(project_image, new_job, user)
    
    return new_job

def assign_uploaded_image_to_batch(
    project_image,
    batch_id: Optional[str],
    user=None,
) -> Optional['Job']:
    """
    Assigns a project image to a batch job.
    
    Logic:
    1. If no batch_id, fall back to auto-job assignment
    2. If batch_id exists, find or create a job with that batch_id
    3. Assign image to that job
    
    Returns the Job it was assigned to.
    """
    # Fallback to auto-job if no batch_id
    if not batch_id:
        return assign_image_to_available_job(project_image, user=user)
    
    # Check if already assigned
    existing_assignment = JobImage.objects.filter(
        project_image=project_image
    ).select_related('job').first()
    
    if existing_assignment:
        logging.info(f"Image already assigned to job {existing_assignment.job.id}")
        return existing_assignment.job

    # Get or create batch job with external lock
    job = _get_or_create_batch_job(project_image.project, batch_id, user)
    
    with transaction.atomic():
        _assign_to_job(project_image, job, user)
    
    return job