from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from uuid import UUID
import uuid as uuid_lib
from typing import Optional
from redis import Redis

from api.dependencies import get_project_context, ProjectContext
from api.routers.suggestions.schemas import *
from api.routers.suggestions.services import SuggestionService
from api.routers.suggestions.segmentation_service import SegmentationService
from api.routers.suggestions.dependencies import get_segmentation_service

router = APIRouter(prefix="/projects")

# Dependency
def get_suggestion_service() -> SuggestionService:
    return SuggestionService()

@router.post(
    "/{project_id}/images/{image_id}/suggestions/segment/text",
    response_model=SuggestionSessionResponse,
    summary="Segment from text prompt (SAM3)"
)
async def segment_from_text(
    project_id: UUID,
    image_id: int,
    request: SAMTextRequest,
    ctx: ProjectContext = Depends(get_project_context),
    seg_svc: SegmentationService = Depends(get_segmentation_service),
    sug_svc: SuggestionService = Depends(get_suggestion_service)
):
    """
    Text prompt → finds all matching objects.
    Example: "person", "red car", "crack in concrete"
    """
    session = await sug_svc.get_session(request.session_id)

    image = await sug_svc.load_image(image_id)
    outputs = seg_svc.segment_from_text(
        image, 
        request.text,
        model=session.model_name,
        device=session.model_device,
        precision=session.model_precision
    )
    
    suggestions = [
        Suggestion(
            suggestion_id=str(uuid_lib.uuid4()),
            bbox=BoundingBox(x=o.bbox[0], y=o.bbox[1],
                             width=o.bbox[2], height=o.bbox[3]),
            mask_rle=o.mask_rle,
            polygons=o.polygons,
            confidence=o.confidence
        )
        for o in outputs
    ]
    
    await sug_svc.store_suggestions(request.session_id, suggestions)
    await sug_svc.update_session_source_type(request.session_id, SuggestionSourceType.SAM_TEXT)
    
    return SuggestionSessionResponse(
        session_id=request.session_id,
        status="ready",
        source_type=SuggestionSourceType.SAM_TEXT,
        suggestions=suggestions,
        count=len(suggestions),
        config=ModelConfig(...)
    )