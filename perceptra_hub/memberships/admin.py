# apps/memberships/admin.py
from django.contrib import admin
from .models import Role, ProjectMembership, OrganizationMembership
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
    list_display = ("id", "user", "project", "role", "organization", "created_at")
    list_filter = ("organization", "role")
    search_fields = ("user__username", "project__name")