
from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import PredictionImageResult, PredictionOverlay

@admin.register(PredictionImageResult)
class PredictionImageResultAdmin(ModelAdmin):
    list_display = (
        'id',
        'model_version',
        'dataset_version',
        'image',
        'created_at',
    )
    list_filter = ('model_version', 'dataset_version')
    search_fields = ('image__image_name', 'model_version__version', 'dataset_version__version_name', "id")
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'


@admin.register(PredictionOverlay)
class PredictionOverlayAdmin(ModelAdmin):
    list_display = (
        'id',
        'prediction_result',
        'class_label',
        'confidence',
        'overlay_type',
        'created_at',
    )
    list_filter = ('overlay_type', 'class_label')
    search_fields = ('class_label', 'prediction_result__image__image_name')
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'
