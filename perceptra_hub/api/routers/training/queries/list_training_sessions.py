from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict
from pydantic import BaseModel
from training.models import TrainingSession
from django.utils.timezone import localtime
from django.db.models import Q
from api.dependencies import get_request_context, RequestContext
from asgiref.sync import sync_to_async

router = APIRouter()


class TrainingSessionSummary(BaseModel):
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
    computeResource: Optional[str]
    triggeredBy: Optional[str]
    currentMetrics: Optional[Dict]
    bestMetrics: Optional[Dict]
    errorMessage: Optional[str]
    # Backward compat
    model_version: Optional[Dict]


class TrainingSessionList(BaseModel):
    total: int
    results: List[TrainingSessionSummary]


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
def _query_sessions(
    organization,
    project_id: Optional[str],
    model_id: Optional[str],
    model_uuid: Optional[str],
    search: Optional[str],
    status: Optional[str],
    limit: int,
    offset: int,
):
    qs = TrainingSession.objects.select_related(
        "model_version__model__project",
        "triggered_by",
    ).filter(
        model_version__model__project__organization=organization
    )

    if project_id:
        qs = qs.filter(model_version__model__project__project_id=project_id)
    if model_uuid:
        qs = qs.filter(model_version__model__model_id=model_uuid)
    elif model_id:
        qs = qs.filter(model_version__model__name__icontains=model_id)
    if search:
        qs = qs.filter(
            Q(model_version__model__name__icontains=search)
            | Q(model_version__model__project__name__icontains=search)
        )
    if status:
        qs = qs.filter(status=status)

    total = qs.count()
    sessions = list(qs.order_by("-created_at")[offset : offset + limit])
    return total, sessions


@router.get("/training-sessions", response_model=TrainingSessionList)
async def list_training_sessions(
    project_id: Optional[str] = Query(None),
    model_id: Optional[str] = Query(None),
    model_uuid: Optional[str] = Query(None, description="Filter by exact model UUID"),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context),
):
    total, sessions = await _query_sessions(
        ctx.organization, project_id, model_id, model_uuid, search, status, limit, offset
    )

    results = []
    for session in sessions:
        mv = session.model_version
        model = mv.model
        project = model.project

        current_metrics = session.current_metrics
        if isinstance(current_metrics, list):
            current_metrics = current_metrics[-1] if current_metrics else None

        triggered_by = None
        if session.triggered_by:
            full_name = (
                f"{session.triggered_by.first_name} {session.triggered_by.last_name}".strip()
            )
            triggered_by = full_name or session.triggered_by.email

        results.append(
            TrainingSessionSummary(
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
                computeResource=session.compute_resource or None,
                triggeredBy=triggered_by,
                currentMetrics=current_metrics if current_metrics else None,
                bestMetrics=session.best_metrics if session.best_metrics else None,
                errorMessage=session.error_message or None,
                model_version={"id": mv.id, "version": mv.version_name},
            )
        )

    return {"total": total, "results": results}
