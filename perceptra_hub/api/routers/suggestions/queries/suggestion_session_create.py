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

@router.post(
    "/{project_id}/images/{image_id}/suggestions/session",
    response_model=SuggestionSessionResponse,
    summary="Initialize AI assistance session"
)
async def create_session(
    project_id: UUID,
    image_id: str,
    request: SessionCreateRequest,
    ctx: ProjectContext = Depends(get_project_context),
    sug_svc: SuggestionService = Depends(get_suggestion_service)
):
    """
    Create session with model selection.
    User chooses model once, uses it for all operations.
    """
    session_id = await sug_svc.create_session(
        project=ctx.project,
        image_id=image_id,
        user=ctx.user,
        config=request.config
    )
    
    return SuggestionSessionResponse(
        session_id=session_id,
        status="ready",
        config=request.config,
        
    )