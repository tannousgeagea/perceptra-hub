from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict
from pydantic import BaseModel
from training.models import TrainingSession
from django.utils import timezone
from api.dependencies import get_request_context, require_permission, RequestContext
from asgiref.sync import sync_to_async

router = APIRouter()

VALID_STATUSES = {"pending", "queued", "initializing", "running", "completed", "failed", "cancelled"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class TrainingUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[float] = None
    logs: Optional[str] = None
    log_path: Optional[str] = None
    metrics: Optional[Dict] = None
    error_message: Optional[str] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None


@sync_to_async
def _apply_update(session_id: int, data: TrainingUpdate, organization):
    try:
        session = TrainingSession.objects.select_related(
            "model_version__model__project__organization"
        ).get(id=session_id)
    except TrainingSession.DoesNotExist:
        raise HTTPException(status_code=404, detail="Training session not found")

    if session.model_version.model.project.organization_id != organization.id:
        raise HTTPException(status_code=403, detail="Access denied to this training session")

    now = timezone.now()

    if data.status:
        if data.status not in VALID_STATUSES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status '{data.status}'. Must be one of: {sorted(VALID_STATUSES)}",
            )
        session.status = data.status
        if data.status == "running" and not session.started_at:
            session.started_at = now
        if data.status in TERMINAL_STATUSES and not session.completed_at:
            session.completed_at = now

    if data.error_message:
        session.error_message = data.error_message
        if session.status not in TERMINAL_STATUSES:
            session.status = "failed"
        if not session.completed_at:
            session.completed_at = now

    if data.progress is not None:
        session.progress = max(0.0, min(100.0, data.progress))
    if data.current_epoch is not None:
        session.current_epoch = data.current_epoch
    if data.total_epochs is not None:
        session.total_epochs = data.total_epochs
    if data.log_path:
        session.log_file_key = data.log_path
    if data.logs:
        session.log_summary = (session.log_summary or "") + data.logs + "\n"
    if data.metrics:
        if not isinstance(session.current_metrics, list):
            session.current_metrics = []
        session.current_metrics.append(data.metrics)
        # Update best_metrics
        if not session.best_metrics:
            session.best_metrics = data.metrics
        else:
            updated_best = dict(session.best_metrics)
            for key, value in data.metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                current_best = updated_best.get(key)
                if current_best is None:
                    updated_best[key] = value
                elif key.startswith("loss") or key.endswith("_loss"):
                    if value < current_best:
                        updated_best[key] = value
                else:
                    if value > current_best:
                        updated_best[key] = value
            session.best_metrics = updated_best

    session.save()
    return session


@router.patch("/training-sessions/{session_id}")
async def update_session(
    session_id: int,
    data: TrainingUpdate,
    ctx: RequestContext = Depends(require_permission("write")),
):
    session = await _apply_update(session_id, data, ctx.organization)
    return {"message": "Training session updated", "status": session.status}
