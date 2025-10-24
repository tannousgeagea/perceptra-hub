# apps/memberships/models.py
from django.db import models
from django.contrib.auth import get_user_model
from projects.models import Project
from organizations.models import Organization
from django.core.exceptions import ValidationError
User = get_user_model()

class Role(models.Model):
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "role"
        verbose_name_plural = "Roles"

    def __str__(self):
        return self.name
    
class OrganizationMembership(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    role = models.ForeignKey(Role, on_delete=models.CASCADE, related_name="members")
    joined_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ("user", "organization")
        db_table = "organization_membership"
        verbose_name_plural = "Organization Memberships"

    def __str__(self):
        return f"{self.user.username} → {self.organization.name} ({self.role.name})"


class ProjectMembership(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="project_memberships")
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="memberships")
    role = models.ForeignKey(Role, on_delete=models.CASCADE, related_name="project_members")
    organization = models.ForeignKey(
        Organization, on_delete=models.SET_NULL, null=True, blank=True, related_name="memberships"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "project")
        db_table = "project_membership"
        verbose_name_plural = "Project Memberships"

    def __str__(self):
        return f"{self.user.username} → {self.project.name} ({self.role.name})"

    def clean(self):
        if self.organization and self.organization != self.project.organization:
            raise ValidationError("Organization must match the project's organization")

    def save(self, *args, **kwargs):
        if not self.organization:
            self.organization = self.project.organization
        self.full_clean()
        super().save(*args, **kwargs)

