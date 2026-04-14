# apps/memberships/models.py
from django.db import models
from django.contrib.auth import get_user_model
from projects.models import Project
from organizations.models import Organization
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
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
    
    # Billing Classification (ADD THESE)
    is_external_annotator = models.BooleanField(
        default=False,
        db_index=True,
        help_text=_('Whether this user is an external contractor in this organization')
    )
    
    billing_enabled = models.BooleanField(
        default=False,
        db_index=True,
        help_text=_('Whether to track billable actions for this user in this org')
    )
    
    billing_rate_card = models.ForeignKey(
        'billing.BillingRateCard',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='assigned_org_members',
        help_text=_('Personal rate card for this user in this organization')
    )
    
    hourly_rate = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text=_('Optional hourly rate for time-based billing')
    )
    
    # Contractor Details
    contractor_company = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text=_('External company/agency this contractor works for')
    )
    
    contract_start_date = models.DateField(
        null=True,
        blank=True,
        help_text=_('When contract started')
    )
    
    contract_end_date = models.DateField(
        null=True,
        blank=True,
        help_text=_('When contract ends (null = ongoing)')
    )

    updated_at = models.DateTimeField(auto_now=True)
    joined_at = models.DateTimeField(auto_now_add=True)
    invited_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='organization_invitations_sent'
    )    

    status = models.CharField(
        max_length=20,
        choices=[
            ('active', 'Active'),
            ('inactive', 'Inactive'),
            ('pending', 'Pending'),
            ('suspended', 'Suspended'),
        ],
        default='active',
        db_index=True,
        help_text="Membership status in this organization"
    )
    
    
    class Meta:
        unique_together = ("user", "organization")
        db_table = "organization_membership"
        verbose_name_plural = "Organization Memberships"
        indexes = [
            models.Index(fields=['user', 'organization']),
            models.Index(fields=['organization', 'role']),
            models.Index(fields=['organization', 'status']),  # ADD THIS
            models.Index(fields=['organization', 'is_external_annotator']),  # NEW
            models.Index(fields=['organization', 'billing_enabled']),  # NEW
        ]

    def __str__(self):
        return f"{self.user.get_full_name() or self.user.username} - {self.organization.name} ({self.role})"
    
    def __repr__(self):
        return f"<OrganizationMember: {self.user.username}@{self.organization.slug}>"

class ProjectMembership(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="project_memberships")
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="memberships")
    role = models.ForeignKey(Role, on_delete=models.CASCADE, related_name="project_members")
    organization = models.ForeignKey(
        Organization, on_delete=models.SET_NULL, null=True, blank=True, related_name="memberships"
    )

    # Project-Specific Billing (ADD THESE)
    is_external_annotator = models.BooleanField(
        default=False,
        db_index=True,
        help_text=_('Whether this user is external contractor for this specific project')
    )
    
    billing_enabled = models.BooleanField(
        default=False,
        db_index=True,
        help_text=_('Whether to track billable actions for this user in this project')
    )
    
    billing_rate_card = models.ForeignKey(
        'billing.BillingRateCard',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='assigned_project_members',
        help_text=_('Project-specific rate card for this user')
    )
    
    hourly_rate = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text=_('Project-specific hourly rate')
    )

    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "project")
        db_table = "project_membership"
        verbose_name_plural = "Project Memberships"
        indexes = [
            models.Index(fields=['project', 'is_external_annotator']),  # NEW
            models.Index(fields=['project', 'billing_enabled']),  # NEW
        ]

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

