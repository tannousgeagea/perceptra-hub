
from django.contrib import admin
from unfold.admin import ModelAdmin, TabularInline
from .models import Image, Tag, ImageTag


@admin.register(Tag)
class TagAdmin(ModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)

class ImageTagInline(TabularInline):
    model = ImageTag
    extra = 1
    autocomplete_fields = ['tag', 'tagged_by']

@admin.register(Image)
class ImageAdmin(ModelAdmin):
    list_display = ('image_name', 'image_file', 'created_at',)
    search_fields = ('image_name', )
    list_filter = ('created_at', 'source_of_origin',)
    inlines = [ImageTagInline]

@admin.register(ImageTag)
class ImageTagAdmin(ModelAdmin):
    list_display = ('image', 'tag', 'tagged_by', 'tagged_at')
    search_fields = ('image__image_name', 'tag__name', 'tagged_by__username')
    autocomplete_fields = ['image', 'tag', 'tagged_by']