from django.contrib import admin
from unfold.admin import ModelAdmin

from notifications.models import Notification


@admin.register(Notification)
class NotificationAdmin(ModelAdmin):
    list_display = ('title', 'kind', 'recipient', 'organization', 'count', 'read_at', 'updated_at')
    list_filter = ('kind', 'organization')
    search_fields = ('title', 'body', 'recipient__username', 'recipient__email')
    raw_id_fields = ('recipient', 'actor', 'project', 'organization')
    readonly_fields = ('created_at', 'updated_at')
