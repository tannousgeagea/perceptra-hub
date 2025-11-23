
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
    "/{project_id}/images/{image_id}/suggestions/sam/auto",
    response_model=SuggestionSessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate SAM auto-segmentation suggestions"
)
async def sam_auto_segment(
    project_id: UUID,
    image_id: int,
    request: SAMAutoRequest,
    background_tasks: BackgroundTasks,
    ctx: ProjectContext = Depends(get_project_context),
    svc: SuggestionService = Depends(get_suggestion_service)
):
    """
    Triggers SAM to auto-segment the image.
    Returns immediately with session_id. Poll GET endpoint for results.
    """
    session_id = await svc.create_session(
        project=ctx.project,
        image_id=image_id,
        source_type=SuggestionSourceType.SAM_AUTO,
        user=ctx.user
    )
    
    background_tasks.add_task(
        svc.run_sam_auto,
        session_id=session_id,
        image_id=image_id,
        config=request
    )
    
    return SuggestionSessionResponse(
        session_id=session_id,
        status="pending",
        source_type=SuggestionSourceType.SAM_AUTO,
    )