
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from uuid import UUID
import uuid as uuid_lib
from typing import Optional
from redis import Redis

from api.dependencies import get_project_context, ProjectContext
from api.routers.suggestions.schemas import *
from api.routers.suggestions.services import SuggestionService

router = APIRouter(prefix="/projects")

# Dependency
# Shared dependency: also requires annotator or above in the project.
from api.routers.suggestions.dependencies import get_suggestion_service  # noqa: E402

@router.delete(
    "/{project_id}/suggestions/session/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear suggestion session"
)
async def clear_session(
    project_id: UUID,
    session_id: UUID,
    ctx: ProjectContext = Depends(get_project_context),
    svc: SuggestionService = Depends(get_suggestion_service)
):
    """Discard all suggestions in session."""
    await svc.get_session(session_id, ctx.project)  # 404 unless it is this project's
    await svc.clear_session(session_id)