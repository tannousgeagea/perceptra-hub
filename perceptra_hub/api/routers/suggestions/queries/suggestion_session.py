
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

@router.get(
    "/{project_id}/suggestions/session/{session_id}",
    response_model=SuggestionSessionResponse,
    summary="Get session info"
)
async def get_session(
    project_id: UUID,
    session_id: UUID,
    ctx: ProjectContext = Depends(get_project_context),
    sug_svc: SuggestionService = Depends(get_suggestion_service)
):
    """Get session with current model config and suggestions."""
    session = await sug_svc.get_session(session_id)
    suggestions = sug_svc._get_suggestions(session_id)
    
    return SuggestionSessionResponse(
        session_id=session_id,
        status="ready",
        config=ModelConfig(
            model=session.model_name,
            device=session.model_device,
            precision=session.model_precision
        ),
        suggestions=[Suggestion(**s) for s in suggestions],
        count=len(suggestions)
    )