
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from uuid import UUID
import uuid as uuid_lib
from typing import Optional
from redis import Redis

from api.dependencies import get_project_context, ProjectContext
from api.routers.suggestions.schemas import *
from api.routers.suggestions.services import SuggestionService
from api.routers.suggestions.segmentation_service import SegmentationService

router = APIRouter(prefix="/projects")

# Dependency
def get_suggestion_service() -> SuggestionService:
    return SuggestionService()

@router.post(
    "/{project_id}/images/{image_id}/suggestions/{session_id}/accept",
    status_code=status.HTTP_201_CREATED,
    summary="Accept suggestions and create annotations"
)
async def accept_suggestions(
    project_id: UUID,
    image_id: int,
    session_id: UUID,
    request: AcceptSuggestionsRequest,
    ctx: ProjectContext = Depends(get_project_context),
    svc: SuggestionService = Depends(get_suggestion_service)
):
    """
    Accept selected suggestions → creates real annotations.
    Uses existing annotation create logic.
    """
    created = await svc.accept_suggestions(
        session_id=session_id,
        suggestion_ids=request.suggestion_ids,
        class_id_override=request.class_id,
        user=ctx.user,
        project=ctx.project,
        image_id=image_id
    )
    
    return {
        "message": f"Created {len(created)} annotations",
        "annotation_uids": created
    }