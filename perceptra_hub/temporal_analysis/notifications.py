"""
Alert notification handlers
"""

from typing import List
from django.conf import settings
import httpx


async def send_slack_notification(alert):
    """Send alert to Slack"""
    
    webhook_url = getattr(settings, 'ALERT_SLACK_WEBHOOK', None)
    if not webhook_url:
        return
    
    color = "#d32f2f" if alert.severity == 'critical' else "#f57c00"
    
    payload = {
        "attachments": [{
            "color": color,
            "title": f"🚨 {alert.severity.upper()}: {alert.metric_name}",
            "text": alert.message,
            "fields": [
                {"title": "Project", "value": alert.project.name, "short": True},
                {"title": "Current Value", "value": f"{alert.current_value:.2%}", "short": True},
            ],
            "footer": "Evaluation Alert System",
        }]
    }
    
    async with httpx.AsyncClient() as client:
        await client.post(webhook_url, json=payload)


async def send_email_notification(alert):
    """Send alert email"""
    
    from django.core.mail import send_mail
    
    subject = f"[{alert.severity.upper()}] {alert.metric_name} Alert - {alert.project.name}"
    
    send_mail(
        subject=subject,
        message=alert.message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[settings.ALERT_EMAIL_RECIPIENT],
        fail_silently=True,
    )