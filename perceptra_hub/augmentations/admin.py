
from django.contrib import admin
from unfold.admin import ModelAdmin, TabularInline
from .models import (
    Augmentation, 
    AugmentationParameter, 
    AugmentationParameterAssignment,
    DatasetAugmentation, 
    DatasetAugmentationParameter,
    VersionImageAugmentation,
)

# ---- Augmentation Parameter Admin ----
@admin.register(AugmentationParameter)
class AugmentationParameterAdmin(ModelAdmin):
    list_display = ("name", "parameter_type", "get_choices")
    search_fields = ("name",)
    list_filter = ("parameter_type",)

    def get_choices(self, obj):
        """Display choices only for 'choice' parameter types"""
        return obj.choices if obj.parameter_type == "choice" else "-"
    get_choices.short_description = "Choices"

# ---- Augmentation Parameter Assignment (Inline) ----
class AugmentationParameterAssignmentInline(TabularInline):
    model = AugmentationParameterAssignment
    extra = 1  # Show empty row for adding new assignments
    min_num = 0
    max_num = 10
    fields = ("parameter", "default_value", "min_value", "max_value")
    autocomplete_fields = ("parameter",)

# ---- Augmentation Admin ----
@admin.register(Augmentation)
class AugmentationAdmin(ModelAdmin):
    list_display = ("name", "description", "is_active")
    search_fields = ("name",)
    list_filter = ("is_active",)
    inlines = [AugmentationParameterAssignmentInline]
    actions = ["activate_augmentations", "deactivate_augmentations"]

    def activate_augmentations(self, request, queryset):
        queryset.update(is_active=True)
    activate_augmentations.short_description = "Activate selected augmentations"

    def deactivate_augmentations(self, request, queryset):
        queryset.update(is_active=False)
    deactivate_augmentations.short_description = "Deactivate selected augmentations"

# ---- Dataset Augmentation Parameter (Inline) ----
class DatasetAugmentationParameterInline(admin.TabularInline):
    model = DatasetAugmentationParameter
    extra = 1
    fields = ("augmentation", "parameter", "value")
    autocomplete_fields = ("augmentation", "parameter")

# ---- Dataset Augmentation Admin ----
@admin.register(DatasetAugmentation)
class DatasetAugmentationAdmin(ModelAdmin):
    list_display = ("version", "is_active", "get_augmentations")
    search_fields = ("version__project__name",)
    list_filter = ("is_active",)
    inlines = [DatasetAugmentationParameterInline]

    def get_augmentations(self, obj):
        return ", ".join([aug.name for aug in obj.augmentations.all()])
    get_augmentations.short_description = "Augmentations"

# ---- Dataset Augmentation Parameter Admin ----
@admin.register(DatasetAugmentationParameter)
class DatasetAugmentationParameterAdmin(ModelAdmin):
    list_display = ("dataset_augmentation", "augmentation", "parameter", "value")
    search_fields = ("dataset_augmentation__version__project__name", "augmentation__name", "parameter__name")
    list_filter = ("augmentation",)

@admin.register(VersionImageAugmentation)
class VersionImageAugmentationAdmin(ModelAdmin):
    list_display = ("version_image", "augmented_image_file")