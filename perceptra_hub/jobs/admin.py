from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import Job, JobImage

@admin.register(Job)
class JobAdmin(ModelAdmin):
    list_display = ('id', 'name', 'project', 'assignee', 'status', 'created_at', 'updated_at')
    list_filter = ('status', 'project', 'assignee')
    search_fields = ('name', 'project__name', 'assignee__username')
    autocomplete_fields = ('project', 'assignee')
    ordering = ('-created_at',)

@admin.register(JobImage)
class JobImageAdmin(ModelAdmin):
    list_display = ('id', 'job', 'project_image', 'created_at')
    search_fields = ('job__name', 'project_image__image__image_name')
    autocomplete_fields = ('job', 'project_image')
    ordering = ('-created_at',)
