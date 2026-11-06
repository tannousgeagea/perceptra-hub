from django.contrib.auth import get_user_model
from django.db import models
from django.utils.translation import gettext_lazy as _

from organizations.models import Organization
from projects.models import Project

User = get_user_model()


class Notification(models.Model):
    """
    One notice for one user in one organization.

    Repeated events of the same kind (a reviewer sending back image after
    image) are folded into the recipient's unread notice with the same
    `group_key`, whose `count` goes up, instead of one row per image.
    """

    class Kind(models.TextChoices):
        ASSIGNED = 'assigned', _('Job assigned')
        SENT_BACK = 'sent_back', _('Work sent back')
        READY_FOR_REVIEW = 'ready_for_review', _('Ready for review')
        COMPLETED = 'completed', _('Job completed')

    recipient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='notifications')
    project = models.ForeignKey(
        Project, on_delete=models.CASCADE, null=True, blank=True, related_name='notifications',
    )
    actor = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='notifications_caused',
        help_text=_('Who caused it, when someone did'),
    )
    kind = models.CharField(max_length=32, choices=Kind.choices)
    title = models.CharField(max_length=255)
    body = models.TextField(blank=True, default='')
    url = models.CharField(max_length=500, blank=True, default='', help_text=_('Console path to open'))
    group_key = models.CharField(
        max_length=128, blank=True, default='', help_text=_('Unread notices with the same key are merged'),
    )
    count = models.PositiveIntegerField(default=1, help_text=_('Events folded into this notice'))
    data = models.JSONField(default=dict, blank=True)
    read_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    class Meta:
        db_table = 'notification'
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['recipient', 'organization', '-updated_at']),
            models.Index(fields=['recipient', 'organization', 'read_at']),
            models.Index(fields=['recipient', 'group_key', 'read_at']),
        ]

    def __str__(self):
        return f"{self.kind} → {self.recipient_id}: {self.title}"

    @property
    def is_read(self) -> bool:
        return self.read_at is not None
