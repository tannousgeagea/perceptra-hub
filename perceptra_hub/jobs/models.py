from django.db import models
from projects.models import Project, ProjectImage
from django.contrib.auth import get_user_model
User = get_user_model()

class Job(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="jobs")
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    assignee = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="assigned_jobs")
    
    STATUS_CHOICES = [
        ("unassigned", "Unassigned"),
        ("assigned", "Assigned"),
        ("in_review", "In Review"),
        ("completed", "Completed"),
        ("sliced", "Sliced"),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="unassigned")
    image_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    batch_id = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        db_table = "job"
        verbose_name_plural = "Jobs"

    def __str__(self):
        return f"{self.project.name} - {self.name}"

class JobImage(models.Model):
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="images")
    project_image = models.ForeignKey(ProjectImage, on_delete=models.CASCADE, related_name="job_assignments")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("job", "project_image")
        db_table = "job_image"
        verbose_name_plural = "Job Images"

    def __str__(self):
        return f"{self.job.name} - {self.project_image}"
