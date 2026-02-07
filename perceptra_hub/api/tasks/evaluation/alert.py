

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

@shared_task(name='evaluation.check_metric_alerts')
def check_metric_alerts(project_id: int, snapshot_id: int):
    """
    Check newly created snapshot for alerts.
    Called after each snapshot creation.
    """
    from temporal_analysis.models import MetricSnapshot, MetricAlert
    from temporal_analysis.alerts import AlertRuleEngine
    
    try:
        # Get current snapshot
        current = MetricSnapshot.objects.get(id=snapshot_id)
        
        # Get previous snapshot (same project, same model if specified)
        previous = MetricSnapshot.objects.filter(
            project=current.project,
            model_version=current.model_version,
            id__lt=current.id
        ).order_by('-snapshot_date').first()
        
        # Evaluate alerts
        engine = AlertRuleEngine()
        alerts_to_create = engine.evaluate_snapshot(current, previous)
        
        created_alerts = []
        for alert_data in alerts_to_create:
            alert = MetricAlert.objects.create(
                project=current.project,
                model_version=current.model_version,
                **alert_data
            )
            created_alerts.append(alert)
        
        # Send notifications if critical alerts
        if any(a.severity == 'critical' for a in created_alerts):
            send_alert_notifications.delay(project_id, [a.id for a in created_alerts])
        
        return {
            'status': 'success',
            'alerts_created': len(created_alerts),
            'alert_ids': [a.id for a in created_alerts]
        }
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}