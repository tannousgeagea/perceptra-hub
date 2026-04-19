
from fastapi import APIRouter, Depends, HTTPException, status
from uuid import UUID
import uuid as uuid_lib

from api.dependencies import get_project_context, ProjectContext
from api.routers.suggestions.schemas import (
    SAMAutoRequest,
    SuggestionSessionResponse,
    SuggestionSourceType,
    ModelConfig,
)
from api.routers.suggestions.services import SuggestionService
from api.routers.suggestions.dependencies import get_suggestion_service

router = APIRouter(prefix="/projects")


@router.post(
    "/{project_id}/images/{image_id}/suggestions/sam/auto",
    response_model=SuggestionSessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate SAM auto-segmentation suggestions",
)
async def sam_auto_segment(
    project_id: UUID,
    image_id: int,
    request: SAMAutoRequest,
    ctx: ProjectContext = Depends(get_project_context),
    svc: SuggestionService = Depends(get_suggestion_service),
):
    """
    Triggers SAM to auto-segment the image.
    Returns 202 immediately. Poll GET /{session_id} for status/results.
    """
    from api.tasks.seg_inference import auto_segment_task
    import os

    session = await svc.get_session(request.session_id)

    if not os.getenv("SEG_INFERENCE_URL", "").strip():
        raise HTTPException(
            status_code=503,
            detail="Inference server not configured — set SEG_INFERENCE_URL",
        )

    auto_segment_task.delay(
        session_id=str(request.session_id),
        image_id=image_id,
        model=session.model_name,
        device=session.model_device,
        precision=session.model_precision,
        config={
            "points_per_side": request.points_per_side,
            "pred_iou_thresh": request.pred_iou_thresh,
            "stability_score_thresh": request.stability_score_thresh,
            "min_area": request.min_area,
        },
    )

    return SuggestionSessionResponse(
        session_id=request.session_id,
        status="pending",
        source_type=SuggestionSourceType.SAM_AUTO,
        config=ModelConfig(
            model=session.model_name,
            device=session.model_device,
            precision=session.model_precision,
        ),
    )
