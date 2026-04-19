from fastapi import APIRouter, Depends, status
from uuid import UUID

from api.dependencies import get_project_context, ProjectContext
from api.routers.suggestions.schemas import RejectSuggestionsRequest
from api.routers.suggestions.services import SuggestionService

router = APIRouter(prefix="/projects")


def get_suggestion_service() -> SuggestionService:
    return SuggestionService()


@router.post(
    "/{project_id}/suggestions/session/{session_id}/reject",
    status_code=status.HTTP_200_OK,
    summary="Reject suggestions",
)
async def reject_suggestions(
    project_id: UUID,
    session_id: UUID,
    request: RejectSuggestionsRequest,
    ctx: ProjectContext = Depends(get_project_context),
    svc: SuggestionService = Depends(get_suggestion_service),
):
    """Remove rejected suggestions from the session cache."""
    await svc.reject_suggestions(
        session_id=session_id,
        suggestion_ids=request.suggestion_ids,
    )
    return {"message": f"Rejected {len(request.suggestion_ids)} suggestions"}
