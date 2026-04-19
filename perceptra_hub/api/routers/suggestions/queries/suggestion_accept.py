from fastapi import APIRouter, Depends, status
from uuid import UUID

from api.dependencies import get_project_context, ProjectContext
from api.routers.suggestions.schemas import AcceptSuggestionsRequest
from api.routers.suggestions.services import SuggestionService

router = APIRouter(prefix="/projects")


def get_suggestion_service() -> SuggestionService:
    return SuggestionService()


@router.post(
    "/{project_id}/suggestions/session/{session_id}/accept",
    status_code=status.HTTP_201_CREATED,
    summary="Accept suggestions and create annotations",
)
async def accept_suggestions(
    project_id: UUID,
    session_id: UUID,
    request: AcceptSuggestionsRequest,
    ctx: ProjectContext = Depends(get_project_context),
    svc: SuggestionService = Depends(get_suggestion_service),
):
    """
    Convert accepted suggestions into real annotations.
    The target image is resolved from the session — no image_id needed in the URL.
    """
    session = await svc.get_session(session_id)

    created = await svc.accept_suggestions(
        session_id=session_id,
        suggestion_ids=request.suggestion_ids,
        class_id_override=request.class_id,
        class_name_override=request.class_name,
        user=ctx.user,
        project=ctx.project,
        image_id=session.project_image_id,
        use_polygon=request.use_polygon,
    )

    return {
        "message": f"Created {len(created)} annotations",
        "annotation_uids": created,
    }
