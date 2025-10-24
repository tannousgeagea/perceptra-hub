from django.contrib import admin
from .models import Organization
from unfold.admin import ModelAdmin
# Register your models here.

@admin.register(Organization)
class OrganizationAdmin(ModelAdmin):
    list_display = ("id", "org_id", "name", "created_at")
    search_fields = ("name",)