"""
Alert Management API Endpoints
"""

from fastapi import APIRouter, Path, Body, HTTPException
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from temporal_analysis.models import MetricAlert
from projects.models import Project
from api.routers.alerts.schemas import AlertResponse, AlertStats, AcknowledgeRequest


router = APIRouter(prefix="/alerts",)


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get(
    "/projects/{project_id}",
    response_model=List[AlertResponse],
    summary="Get alerts for a project"
)
async def get_project_alerts(
    project_id: int = Path(...),
    acknowledged: Optional[bool] = None,
    severity: Optional[str] = None,
    limit: int = 100,
):
    """
    Get alerts for a project.
    Filter by acknowledged status and severity.
    """
    
    # Verify project
    try:
        project = Project.objects.get(id=project_id, is_deleted=False)
    except Project.DoesNotExist:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Build query
    alerts = MetricAlert.objects.filter(
        project_id=project_id
    ).select_related('project', 'acknowledged_by')
    
    if acknowledged is not None:
        alerts = alerts.filter(is_acknowledged=acknowledged)
    
    if severity:
        alerts = alerts.filter(severity=severity)
    
    alerts = alerts.order_by('-alert_date')[:limit]
    
    # Transform response
    return [
        AlertResponse(
            id=a.id,
            project_id=a.project_id,
            project_name=a.project.name,
            severity=a.severity,
            metric_name=a.metric_name,
            current_value=a.current_value,
            previous_value=a.previous_value,
            threshold_value=a.threshold_value,
            change_percent=a.change_percent,
            message=a.message,
            created_at=a.alert_date,
            is_acknowledged=a.is_acknowledged,
            acknowledged_at=a.acknowledged_at,
            acknowledged_by=a.acknowledged_by.username if a.acknowledged_by else None
        )
        for a in alerts
    ]


@router.get(
    "/projects/{project_id}/stats",
    response_model=AlertStats,
    summary="Get alert statistics"
)
async def get_alert_stats(project_id: int = Path(...)):
    """Get alert statistics for a project"""
    
    from django.db.models import Count, Q
    
    stats = MetricAlert.objects.filter(
        project_id=project_id
    ).aggregate(
        total=Count('id'),
        critical=Count('id', filter=Q(severity='critical')),
        warning=Count('id', filter=Q(severity='warning')),
        unacknowledged=Count('id', filter=Q(is_acknowledged=False)),
        acknowledged=Count('id', filter=Q(is_acknowledged=True))
    )
    
    return AlertStats(
        total_alerts=stats['total'],
        critical_count=stats['critical'],
        warning_count=stats['warning'],
        unacknowledged_count=stats['unacknowledged'],
        acknowledged_count=stats['acknowledged']
    )


@router.post(
    "/{alert_id}/acknowledge",
    response_model=AlertResponse,
    summary="Acknowledge an alert"
)
async def acknowledge_alert(
    alert_id: int = Path(...),
    request: AcknowledgeRequest = Body(...),
):
    """Mark an alert as acknowledged"""
    
    from django.contrib.auth import get_user_model
    from django.utils import timezone
    
    User = get_user_model()
    
    try:
        alert = MetricAlert.objects.select_related('project').get(id=alert_id)
    except MetricAlert.DoesNotExist:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    # Get user
    try:
        user = User.objects.get(username=request.acknowledged_by)
    except User.DoesNotExist:
        # If user not found, just store username as string
        user = None
    
    # Update alert
    alert.is_acknowledged = True
    alert.acknowledged_at = timezone.now()
    alert.acknowledged_by = user
    alert.save()
    
    return AlertResponse(
        id=alert.id,
        project_id=alert.project_id,
        project_name=alert.project.name,
        severity=alert.severity,
        metric_name=alert.metric_name,
        current_value=alert.current_value,
        previous_value=alert.previous_value,
        threshold_value=alert.threshold_value,
        change_percent=alert.change_percent,
        message=alert.message,
        created_at=alert.alert_date,
        is_acknowledged=True,
        acknowledged_at=alert.acknowledged_at,
        acknowledged_by=request.acknowledged_by
    )