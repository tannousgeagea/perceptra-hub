import os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from training.models import TrainingSession
from api.dependencies import get_request_context, RequestContext
from asgiref.sync import sync_to_async

router = APIRouter()


@sync_to_async
def _get_session(session_id: int, ctx: RequestContext):
    from projects.dependencies import require_project_permission

    organization = ctx.organization
    try:
        session = TrainingSession.objects.select_related(
            "model_version__model__project__organization"
        ).get(id=session_id)
    except TrainingSession.DoesNotExist:
        raise HTTPException(status_code=404, detail="Training session not found")

    if session.model_version.model.project.organization_id != organization.id:
        raise HTTPException(status_code=403, detail="Access denied to this training session")
    require_project_permission(ctx, session.model_version.model.project, "view", "Viewing this training run")

    return session


@router.get("/training-sessions/{session_id}/logs/stream")
async def stream_logs(
    session_id: int,
    ctx: RequestContext = Depends(get_request_context),
):
    session = await _get_session(session_id, ctx)

    # Prefer storage-backed log file key; fall back to local path convention
    if session.log_file_key:
        log_path = session.log_file_key
    else:
        log_path = f"/media/session_{session_id}.log"

    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    def iterfile():
        with open(log_path, "r") as f:
            for line in f:
                yield line

    return StreamingResponse(iterfile(), media_type="text/plain")
