from django.db import models

# Create your models here.
import uuid
from django.contrib.auth.models import AbstractUser, PermissionsMixin
from django.db import models
from django.utils import timezone

from django.utils.translation import gettext_lazy as _
from django.conf import settings
from .managers import CustomUserManager

class CustomUser(AbstractUser):
    username = models.CharField(max_length=255, unique=True, null=True, blank=True)
    email = models.EmailField(_("email address"), unique=True)
    name = models.CharField(max_length=100, null=True, blank=True)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def __str__(self):
        return self.email

    class Meta:
        db_table = "custom_user"
        verbose_name = _("user")
        verbose_name_plural = _("users")
        constraints = [
            models.UniqueConstraint(fields=['email'], name='unique_email_constraint')
        ]