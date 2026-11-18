
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
# Shared dependency: also requires annotator or above in the project.
from api.routers.suggestions.dependencies import get_suggestion_service  # noqa: E402

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

    Not available yet: the label classifier isn't wired up, and the code that
    was here crashed on every call. The console doesn't use this route.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Label suggestions are not available yet",
    )