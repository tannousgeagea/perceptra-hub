"""
POST /api/v1/training-sessions/{id}/cancel — stop a training run.

Who: whoever may start a run in the project (editor role or above).
409 when the run already ended. See training/cancel.py for how each kind of
compute is stopped.
"""
from typing import Optional

from asgiref.sync import sync_to_async
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import RequestContext, get_request_context

router = APIRouter()


class CancelResponse(BaseModel):
    status: str
    # True: compute accepted the stop; False: couldn't reach it; None: nothing ran yet.
    compute_stopped: Optional[bool] = None
    # "now" | "after_epoch" (agents stop when their current epoch ends)
    stops: str
    detail: str = ""


@sync_to_async
def _cancel(session_id: int, ctx: RequestContext) -> CancelResponse:
    from projects.dependencies import require_project_permission
    from training.cancel import NotActive, cancel_run
    from training.models import TrainingSession

    session = (TrainingSession.objects.select_related("model_version__model__project")
               .filter(id=session_id).first())
    if session is None or session.model_version.model.project.organization_id != ctx.organization.id:
        raise HTTPException(status_code=404, detail="Training session not found")
    require_project_permission(ctx, session.model_version.model.project, "edit", "Stopping a training run")
    try:
        outcome = cancel_run(session, ctx.user)
    except NotActive as exc:
        raise HTTPException(status_code=409, detail=f"This run already ended ({exc}).")
    return CancelResponse(**outcome.__dict__)


@router.post("/training-sessions/{session_id}/cancel", response_model=CancelResponse)
async def cancel_session(session_id: int, ctx: RequestContext = Depends(get_request_context)):
    return await _cancel(session_id, ctx)
