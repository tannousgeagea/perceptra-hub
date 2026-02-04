

import asyncio
import time
from celery import shared_task
from datetime import datetime

# In api/tasks.py

@shared_task(name='evaluation.send_alert_notifications')
def send_alert_notifications(project_id: int, alert_ids: List[int]):
    """Send notifications for alerts"""
    from temporal_analysis.models import MetricAlert
    from temporal_analysis.notifications import send_slack_notification, send_email_notification
    import asyncio
    
    alerts = MetricAlert.objects.filter(id__in=alert_ids)
    
    async def send_all():
        for alert in alerts:
            await send_slack_notification(alert)
            await send_email_notification(alert)
    
    asyncio.run(send_all())
    
    return {'status': 'sent', 'count': len(alerts)}

