
import django
django.setup()
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.conf import settings
from users.models import CustomUser as User

from organizations.models import Organization
from memberships.models import OrganizationMembership, Role

def send_new_user_email(user, password):
    subject = "Welcome to Our Labeling Service [VisionNest]!"
    context = {
        'user': user,
        'password': password,
    }
    message = render_to_string('emails/user_welcome_email.txt', context)
    html_message = render_to_string('emails/user_welcome_email.html', context)
    send_mail(
        subject,
        message,
        settings.DEFAULT_FROM_EMAIL,
        [user.email],
        html_message=html_message,
    )


def send_password_reset_email(user, reset_url: str, valid_hours: int = 1):
    """Email the link to choose a new password. Raises if sending fails."""
    full_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    context = {
        'name': full_name or user.username or user.email,
        'email': user.email,
        'reset_url': reset_url,
        'valid_hours': valid_hours,
    }
    send_mail(
        "Reset your Malumetrix password",
        render_to_string('emails/password_reset_email.txt', context),
        settings.DEFAULT_FROM_EMAIL,
        [user.email],
        html_message=render_to_string('emails/password_reset_email.html', context),
    )




