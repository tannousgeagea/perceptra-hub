# apps/memberships/admin.py
from django.contrib import admin
from .models import Role, ProjectMembership, OrganizationMembership, OrganizationInvitation
from unfold.admin import ModelAdmin

@admin.register(Role)
class RoleAdmin(ModelAdmin):
    list_display = ("id", "name", "description")
    search_fields = ("name",)


@admin.register(OrganizationMembership)
class OrganizationMembershipAdmin(ModelAdmin):
    list_display = ("id", "user", "role", "organization", "joined_at")
    list_filter = ("organization", "role")
    search_fields = ("user__username", "organization__name")

@admin.register(ProjectMembership)
class ProjectMembershipAdmin(ModelAdmin):
    list_display = ("id", "user", "project", "role", "organization", "joined_at")
    list_filter = ("organization", "role")
    search_fields = ("user__username", "project__name")


@admin.register(OrganizationInvitation)
class OrganizationInvitationAdmin(ModelAdmin):
    list_display = ("id", "email", "organization", "role", "status", "invited_by", "created_at", "expires_at")
    list_filter = ("organization", "status", "role")
    search_fields = ("email", "organization__name")
    readonly_fields = ("token", "created_at", "last_sent_at", "send_count", "accepted_at", "accepted_by")