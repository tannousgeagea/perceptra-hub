

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

@router.patch(
    "/{project_id}/suggestions/session/{session_id}/model",
    response_model=SuggestionSessionResponse,
    summary="Change model for session"
)
async def update_session_model(
    project_id: UUID,
    session_id: UUID,
    request: SessionCreateRequest,
    ctx: ProjectContext = Depends(get_project_context),
    sug_svc: SuggestionService = Depends(get_suggestion_service)
):
    """
    Switch to different model mid-session.
    Clears existing suggestions.
    """
    await sug_svc.update_session_model(session_id, request.config)
    await sug_svc.clear_session(session_id)
    
    return SuggestionSessionResponse(
        session_id=session_id,
        status="ready",
        config=request.config,
    )