
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
    "/{project_id}/images/{image_id}/suggestions/labels",
    response_model=SuggestionSessionResponse,
    summary="Suggest labels for a region"
)
async def suggest_labels(
    project_id: UUID,
    image_id: int,
    request: LabelSuggestionRequest,
    ctx: ProjectContext = Depends(get_project_context),
    svc: SuggestionService = Depends(get_suggestion_service)
):
    """
    Given a bbox, suggest most likely class labels.
    Uses CLIP or project-specific classifier.
    """
    session_id = await svc.create_session(
        project=ctx.project,
        image_id=image_id,
        source_type=SuggestionSourceType.LABEL_SUGGEST,
        user=ctx.user
    )
    
    suggestions = await svc.suggest_labels(
        session_id=session_id,
        project=ctx.project,
        image_id=image_id,
        bbox=request.bbox,
        top_k=request.top_k
    )
    
    return SuggestionSessionResponse(
        session_id=session_id,
        status="ready",
        source_type=SuggestionSourceType.LABEL_SUGGEST,
        suggestions=suggestions
    )