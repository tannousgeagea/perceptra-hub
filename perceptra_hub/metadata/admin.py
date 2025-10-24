from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import Language

# Register your models here.
@admin.register(Language)
class LanguageAdmin(ModelAdmin):
    list_display = ('code', 'name', 'created_at')
    search_fields = ('code', 'name')
    ordering = ('-created_at',)