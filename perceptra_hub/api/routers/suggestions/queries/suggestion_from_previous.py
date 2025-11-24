
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
    "/{project_id}/images/{image_id}/suggestions/propagate",
    response_model=SuggestionSessionResponse,
    summary="Propagate annotations from previous image"
)
async def propagate_from_previous(
    project_id: UUID,
    image_id: int,
    request: PreviousFrameRequest,
    ctx: ProjectContext = Depends(get_project_context),
    svc: SuggestionService = Depends(get_suggestion_service)
):
    """
    Copy annotations from source image as suggestions.
    Optionally filter by specific annotation UIDs.
    """
    session_id = await svc.create_session(
        project=ctx.project,
        image_id=image_id,
        source_type=SuggestionSourceType.PREVIOUS_FRAME,
        user=ctx.user,
        source_image_id=request.source_image_id
    )
    
    suggestions = await svc.propagate_annotations(
        session_id=session_id,
        target_image_id=image_id,
        source_image_id=request.source_image_id,
        annotation_uids=request.annotation_uids
    )
    
    return SuggestionSessionResponse(
        session_id=session_id,
        status="ready",
        source_type=SuggestionSourceType.PREVIOUS_FRAME,
        suggestions=suggestions,
        count=len(suggestions),
    )
