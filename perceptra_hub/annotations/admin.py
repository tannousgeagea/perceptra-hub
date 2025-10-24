from django.contrib import admin
from unfold.admin import ModelAdmin, TabularInline
from .models import (
    AnnotationGroup, 
    AnnotationClass,
    AnnotationType,
    Annotation,
    AnnotationAudit,
)

class AnnotationClassInline(TabularInline):
    model = AnnotationClass
    extra =  1

# Register your models here.
@admin.register(AnnotationGroup)
class AnnotationGroupAdmin(ModelAdmin):
    list_display = ('id', 'name', 'project', 'created_at')
    search_fields = ('name', 'project__name')
    list_filter = ('project', 'created_at')
    ordering = ('-created_at',)
    readonly_fields = ('created_at',)
    inlines = [AnnotationClassInline]
    
@admin.register(AnnotationClass)
class AnnotationClassAdmin(ModelAdmin):
    list_display = ('id', 'class_id', 'name', 'annotation_group', 'color', 'created_at')
    search_fields = ('class_id', 'name', 'annotation_group__name', 'color')
    list_filter = ('annotation_group',)
    ordering = ('annotation_group', 'class_id')
    readonly_fields = ('created_at',)

@admin.register(AnnotationType)
class AnnotationTypeAdmin(ModelAdmin):
    list_display = ('id', 'name', 'description', 'created_at')
    search_fields = ('name', 'description')
    list_filter = ('created_at',)
    ordering = ('-created_at',)
    readonly_fields = ('created_at',)

@admin.register(Annotation)
class AnnotationAdmin(ModelAdmin):
    list_display = ('id', 'project_image', 'annotation_class', 'annotation_type','created_at', 'confidence', 'reviewed', 'feedback_provided')
    search_fields = ('project_image__image__image_name', 'annotation_class__name', 'annotation_type__name', 'created_by')
    list_filter = ('annotation_class', 'annotation_type', 'created_at', 'project_image__project__name', 'reviewed', 'feedback_provided')
    ordering = ('-created_at',)
    readonly_fields = ('created_at',)

@admin.register(AnnotationAudit)
class AnnotationAuditAdmin(ModelAdmin):
    list_display = ("id", "annotation", "evaluation_status", "was_edited", "reviewed_at")
    list_filter = ("was_edited", "evaluation_status")
    search_fields = ("id", "annotation_id")