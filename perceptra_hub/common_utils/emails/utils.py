
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




