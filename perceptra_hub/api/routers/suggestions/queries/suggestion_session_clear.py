
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
def get_suggestion_service() -> SuggestionService:
    return SuggestionService()

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
    await svc.clear_session(session_id)