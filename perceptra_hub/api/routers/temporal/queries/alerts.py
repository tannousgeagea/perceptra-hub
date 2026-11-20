"""
Metric Alerts and Thresholds API

GET  /temporal/projects/{project_id}/alerts
POST /temporal/projects/{project_id}/alerts/{alert_id}/acknowledge
GET  /temporal/projects/{project_id}/thresholds
PUT  /temporal/projects/{project_id}/thresholds

project_id is the project's UUID. Viewing needs project membership (or org
admin), like the rest of Evaluation; acknowledging needs the reviewer role;
changing thresholds needs the manager role (they decide when alerts fire,
and alerts can trigger retraining).
"""

from datetime import datetime
from typing import Dict, List, Optional

from asgiref.sync import sync_to_async
from django.utils import timezone
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel

from projects.context import ProjectContext
from projects.dependencies import get_project_context
from temporal_analysis.alerts import (
    AlertRuleEngine,
    save_project_thresholds,
    thresholds_for_project,
    validate_thresholds,
)
from temporal_analysis.models import MetricAlert

router = APIRouter(prefix="/temporal")


class AlertResponse(BaseModel):
    id: int
    severity: str
    metric_name: str
    message: str
    current_value: float
    previous_value: Optional[float] = None
    threshold_value: float
    change_percent: Optional[float] = None
    is_acknowledged: bool
    alert_date: datetime
    acknowledged_at: Optional[datetime] = None


class AcknowledgeResponse(BaseModel):
    id: int
    is_acknowledged: bool


class ThresholdResponse(BaseModel):
    metric: str
    warning_threshold: float
    critical_threshold: float
    higher_is_better: bool
    condition: str  # 'lt' or 'gt'
    # The platform default for this metric (shown next to the project's value).
    default_warning: float
    default_critical: float


class ThresholdsOut(BaseModel):
    thresholds: List[ThresholdResponse]
    # Where alert notifications go (server-wide settings, not per project).
    notify_slack: bool
    notify_email: bool


class ThresholdValue(BaseModel):
    warning: float
    critical: float


class ThresholdsIn(BaseModel):
    # {metric: {warning, critical}}; metrics left out use the defaults.
    thresholds: Dict[str, ThresholdValue]


@router.get(
    "/projects/{project_id}/alerts",
    response_model=List[AlertResponse],
    summary="List metric alerts for a project",
)
async def list_project_alerts(
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgment status"),
    limit: int = Query(50, ge=1, le=200),
    project_ctx: ProjectContext = Depends(get_project_context),
):
    @sync_to_async
    def _alerts():
        qs = MetricAlert.objects.filter(project=project_ctx.project).order_by("-alert_date")
        if acknowledged is not None:
            qs = qs.filter(is_acknowledged=acknowledged)
        return list(qs[:limit])

    return [
        AlertResponse(
            id=a.id,
            severity=a.severity,
            metric_name=a.metric_name,
            message=a.message,
            current_value=a.current_value,
            previous_value=a.previous_value,
            threshold_value=a.threshold_value,
            change_percent=a.change_percent,
            is_acknowledged=a.is_acknowledged,
            alert_date=a.alert_date,
            acknowledged_at=a.acknowledged_at,
        )
        for a in await _alerts()
    ]


@router.post(
    "/projects/{project_id}/alerts/{alert_id}/acknowledge",
    response_model=AcknowledgeResponse,
    summary="Acknowledge a metric alert",
)
async def acknowledge_alert(
    alert_id: int = Path(...),
    project_ctx: ProjectContext = Depends(get_project_context),
):
    project_ctx.require_review_permission()

    @sync_to_async
    def _acknowledge():
        try:
            alert = MetricAlert.objects.get(id=alert_id, project=project_ctx.project)
        except MetricAlert.DoesNotExist:
            raise HTTPException(status_code=404, detail="Alert not found")
        alert.is_acknowledged = True
        alert.acknowledged_at = timezone.now()
        alert.acknowledged_by = project_ctx.effective_user
        alert.save(update_fields=["is_acknowledged", "acknowledged_at", "acknowledged_by"])

    await _acknowledge()
    return AcknowledgeResponse(id=alert_id, is_acknowledged=True)


def _thresholds_out(project) -> ThresholdsOut:
    from temporal_analysis.notifications import email_recipient, slack_configured

    defaults = {t.metric_name: t for t in AlertRuleEngine.DEFAULT_THRESHOLDS}
    return ThresholdsOut(
        thresholds=[
            ThresholdResponse(
                metric=t.metric_name,
                warning_threshold=t.warning_threshold,
                critical_threshold=t.critical_threshold,
                higher_is_better=t.higher_is_better,
                condition="lt" if t.higher_is_better else "gt",
                default_warning=defaults[t.metric_name].warning_threshold,
                default_critical=defaults[t.metric_name].critical_threshold,
            )
            for t in thresholds_for_project(project)
        ],
        notify_slack=slack_configured(),
        notify_email=bool(email_recipient()),
    )


@router.get(
    "/projects/{project_id}/thresholds",
    response_model=ThresholdsOut,
    summary="Get the project's alert thresholds",
)
async def get_thresholds(
    project_ctx: ProjectContext = Depends(get_project_context),
):
    return _thresholds_out(project_ctx.project)


@router.put(
    "/projects/{project_id}/thresholds",
    response_model=ThresholdsOut,
    summary="Save the project's alert thresholds",
)
async def update_thresholds(
    data: ThresholdsIn,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    """Values equal to the defaults are not stored, so later default changes still reach them."""
    project_ctx.require_manage_permission()
    try:
        clean = validate_thresholds({k: v.model_dump() for k, v in data.thresholds.items()})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    defaults = {t.metric_name: t for t in AlertRuleEngine.DEFAULT_THRESHOLDS}
    custom = {
        metric: pair for metric, pair in clean.items()
        if (pair["warning"], pair["critical"])
        != (defaults[metric].warning_threshold, defaults[metric].critical_threshold)
    }

    @sync_to_async
    def _save():
        save_project_thresholds(project_ctx.project.pk, custom)
        project_ctx.project.refresh_from_db(fields=["settings"])

    await _save()
    return _thresholds_out(project_ctx.project)
