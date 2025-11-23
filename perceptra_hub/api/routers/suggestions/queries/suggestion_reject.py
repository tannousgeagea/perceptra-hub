
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
    "/{project_id}/images/{image_id}/suggestions/{session_id}/reject",
    status_code=status.HTTP_200_OK,
    summary="Reject suggestions"
)
async def reject_suggestions(
    project_id: UUID,
    image_id: int,
    session_id: UUID,
    request: RejectSuggestionsRequest,
    ctx: ProjectContext = Depends(get_project_context),
    svc: SuggestionService = Depends(get_suggestion_service)
):
    """Mark suggestions as rejected (for analytics)."""
    await svc.reject_suggestions(
        session_id=session_id,
        suggestion_ids=request.suggestion_ids
    )
    return {"message": f"Rejected {len(request.suggestion_ids)} suggestions"}

