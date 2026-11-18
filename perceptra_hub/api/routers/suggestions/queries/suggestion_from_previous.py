from fastapi import APIRouter, Depends
from uuid import UUID

from api.dependencies import get_project_context, ProjectContext
from api.routers.suggestions.schemas import (
    PreviousFrameRequest,
    SegmentationResponse,
    SuggestionSourceType,
)
from api.routers.suggestions.services import SuggestionService

router = APIRouter(prefix="/projects")


# Shared dependency: also requires annotator or above in the project.
from api.routers.suggestions.dependencies import get_suggestion_service  # noqa: E402


@router.post(
    "/{project_id}/images/{image_id}/suggestions/propagate",
    response_model=SegmentationResponse,
    summary="Propagate annotations from a previous image",
)
async def propagate_from_previous(
    project_id: UUID,
    image_id: int,
    request: PreviousFrameRequest,
    ctx: ProjectContext = Depends(get_project_context),
    svc: SuggestionService = Depends(get_suggestion_service),
):
    """
    Copy active annotations from a source image as pending suggestions on the
    target image.  No AI model is required — this is a pure database operation.

    If the caller has an active SAM session open (session_id provided), the new
    suggestions are appended to that session's cache so they can be
    accepted/rejected alongside SAM suggestions.  If no session_id is given the
    suggestions are returned directly without being persisted in a cache.
    """
    suggestions = await svc.propagate_annotations(
        session_id=request.session_id,
        target_image_id=image_id,
        source_image_id=request.source_image_id,
        annotation_uids=request.annotation_uids,
        project=ctx.project,
    )

    return SegmentationResponse(suggestions=suggestions, count=len(suggestions))
