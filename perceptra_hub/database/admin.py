
# Register your models here.
from django.contrib import admin
# from .models import ProjectType, ImageMode, Project, Image, Annotation

# @admin.register(ProjectType)
# class ProjectTypeAdmin(admin.ModelAdmin):
#     list_display = ('project_type', 'description', 'created_at')
#     search_fields = ('project_type', 'description')
#     list_filter = ('created_at',)

# @admin.register(ImageMode)
# class ImageModeAdmin(admin.ModelAdmin):
#     list_display = ('mode', 'description', 'created_at')
#     search_fields = ('mode', 'description')
#     list_filter = ('created_at',)

# @admin.register(Project)
# class ProjectAdmin(admin.ModelAdmin):
#     list_display = ('project_name', 'project_type', 'annotation_group', 'description', 'created_at')
#     search_fields = ('project_name', 'annotation_group', 'description', 'project_type__project_type')
#     list_filter = ('created_at', 'project_type')

# @admin.register(Image)
# class ImageAdmin(admin.ModelAdmin):
#     list_display = ('image_name', 'image_file', 'created_at', 'annotated', 'processed', 'mode')
#     search_fields = ('image_name', 'mode__mode')
#     list_filter = ('annotated', 'processed', 'mode', 'created_at')
    
# @admin.register(Annotation)
# class AnnotationAdmin(admin.ModelAdmin):
#     list_display = ('project', 'image', 'annotation_file', 'created_at')
#     search_fields = ('project__project_name', 'image__image_name')
#     list_filter = ('created_at', 'project')