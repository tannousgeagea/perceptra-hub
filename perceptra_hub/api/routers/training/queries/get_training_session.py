from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List, Dict
from pydantic import BaseModel
from django.utils.timezone import localtime
from training.models import TrainingSession, TrainingCheckpoint
from training.epochs import best_epoch, epoch_seconds, stalled_for
from api.dependencies import get_request_context, RequestContext
from asgiref.sync import sync_to_async

router = APIRouter()


class EpochMetric(BaseModel):
    epoch: int
    timestamp: str
    is_best: bool
    metrics: Dict[str, float]


class TrainingSessionOut(BaseModel):
    id: str
    session_id: str
    modelName: str
    projectName: str
    modelVersionId: str
    modelVersionName: str
    status: str
    createdAt: str
    updatedAt: str
    startedAt: Optional[str]
    completedAt: Optional[str]
    progress: float
    currentEpoch: int
    totalEpochs: int
    duration: Optional[str]
    estimatedTimeRemaining: Optional[str]
    computeResource: Optional[str]
    triggeredBy: Optional[str]
    currentMetrics: Optional[Dict]
    bestMetrics: Optional[Dict]
    configuration: Optional[Dict]
    logs: List[str]
    errorMessage: Optional[str]
    errorTraceback: Optional[str]
    epochMetrics: List[EpochMetric]
    # Live view
    etaSeconds: Optional[float] = None
    secondsPerEpoch: Optional[float] = None
    lastEpochAt: Optional[str] = None
    # Set while running when no epoch arrived for unusually long.
    stalledSeconds: Optional[float] = None
    # The best epoch on the headline metric (mAP if present, else a loss).
    bestEpoch: Optional[Dict] = None
    # For "Compare with production" once it finishes.
    projectId: Optional[str] = None
    modelId: Optional[str] = None
    versionStatus: Optional[str] = None
    productionVersionId: Optional[str] = None
    productionVersionName: Optional[str] = None
    # Keep for backward compat with frontend hooks
    model_version: Optional[Dict]


def _format_duration(delta) -> Optional[str]:
    if not delta:
        return None
    total_secs = int(delta.total_seconds())
    if total_secs < 0:
        return None
    h, rem = divmod(total_secs, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


@sync_to_async
def _get_session(session_id: int, organization):
    try:
        return TrainingSession.objects.select_related(
            "model_version__model__project__organization",
            "triggered_by",
        ).get(id=session_id)
    except TrainingSession.DoesNotExist:
        raise HTTPException(status_code=404, detail="Training session not found")


@sync_to_async
def _check_org_access(session, organization):
    if session.model_version.model.project.organization_id != organization.id:
        raise HTTPException(status_code=403, detail="Access denied to this training session")


@sync_to_async
def _get_checkpoints(session):
    return list(
        TrainingCheckpoint.objects.filter(training_session=session).order_by("epoch")
    )


# The metric the "best epoch" is judged on, first one present wins.
HEADLINE_KEYS = ("mAP50", "mAP50-95", "val_loss", "train_loss")


@sync_to_async
def _production(model, version_id):
    prod = (model.versions.filter(is_deleted=False, deployment_status="production")
            .exclude(version_id=version_id).order_by("-deployed_at").first())
    return prod


@router.get("/training-sessions/{session_id}")
async def get_training_session(
    session_id: int,
    ctx: RequestContext = Depends(get_request_context),
) -> TrainingSessionOut:
    session = await _get_session(session_id, ctx.organization)
    await _check_org_access(session, ctx.organization)
    checkpoints = await _get_checkpoints(session)

    mv = session.model_version
    model = mv.model
    project = model.project

    # Epochs: the recorded history first, then checkpoints, then the old list form.
    epoch_metrics: List[EpochMetric] = []
    history = session.epoch_history or []
    headline = next((k for k in HEADLINE_KEYS if any(k in (e.get("metrics") or {}) for e in history)), None)
    best = best_epoch(session, headline) if headline else None
    if history:
        for e in history:
            epoch_metrics.append(
                EpochMetric(
                    epoch=e.get("epoch") or 0,
                    timestamp=e.get("at") or localtime(session.updated_at).isoformat(),
                    is_best=best is not None and e.get("epoch") == best.get("epoch"),
                    metrics=e.get("metrics") or {},
                )
            )
    elif checkpoints:
        for cp in checkpoints:
            epoch_metrics.append(
                EpochMetric(
                    epoch=cp.epoch,
                    timestamp=localtime(cp.created_at).isoformat(),
                    is_best=cp.is_best,
                    metrics={
                        k: float(v)
                        for k, v in (cp.metrics or {}).items()
                        if isinstance(v, (int, float))
                    },
                )
            )
    elif isinstance(session.current_metrics, list):
        for i, met in enumerate(session.current_metrics):
            epoch_metrics.append(
                EpochMetric(
                    epoch=i + 1,
                    timestamp=localtime(session.updated_at).isoformat(),
                    is_best=False,
                    metrics={
                        k: float(v)
                        for k, v in met.items()
                        if k != "epoch" and isinstance(v, (int, float))
                    },
                )
            )

    stalled = stalled_for(session)
    production = await _production(model, mv.version_id)

    # Resolve current metrics (latest value if list)
    current_metrics = session.current_metrics
    if isinstance(current_metrics, list):
        current_metrics = current_metrics[-1] if current_metrics else None

    # Triggered-by display name
    triggered_by = None
    if session.triggered_by:
        full_name = (
            f"{session.triggered_by.first_name} {session.triggered_by.last_name}".strip()
        )
        triggered_by = full_name or session.triggered_by.email

    return TrainingSessionOut(
        id=str(session.id),
        session_id=session.session_id,
        modelName=model.name,
        projectName=project.name,
        modelVersionId=str(mv.version_id),
        modelVersionName=mv.version_name,
        status=session.status,
        createdAt=localtime(session.created_at).isoformat(),
        updatedAt=localtime(session.updated_at).isoformat(),
        startedAt=localtime(session.started_at).isoformat() if session.started_at else None,
        completedAt=localtime(session.completed_at).isoformat() if session.completed_at else None,
        progress=session.progress,
        currentEpoch=session.current_epoch,
        totalEpochs=session.total_epochs,
        duration=_format_duration(session.duration),
        estimatedTimeRemaining=_format_duration(session.estimated_time_remaining),
        computeResource=session.compute_resource or None,
        triggeredBy=triggered_by,
        currentMetrics=current_metrics if current_metrics else None,
        bestMetrics=session.best_metrics if session.best_metrics else None,
        configuration=session.config if session.config else None,
        logs=session.log_summary.splitlines() if session.log_summary else [],
        errorMessage=session.error_message or None,
        errorTraceback=session.error_traceback or None,
        epochMetrics=epoch_metrics,
        etaSeconds=(session.estimated_time_remaining.total_seconds()
                    if session.estimated_time_remaining and session.status == "running" else None),
        secondsPerEpoch=epoch_seconds(session),
        lastEpochAt=history[-1].get("at") if history else None,
        stalledSeconds=stalled.total_seconds() if stalled else None,
        bestEpoch=({"epoch": best["epoch"], "metric": headline, "value": best["metrics"][headline]}
                   if best else None),
        projectId=str(project.project_id),
        modelId=model.model_id,
        versionStatus=mv.status,
        productionVersionId=production.version_id if production else None,
        productionVersionName=(production.version_name or f"v{production.version_number}") if production else None,
        model_version={"id": mv.id, "version": mv.version_name},
    )
