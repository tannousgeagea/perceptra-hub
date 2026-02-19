
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
    "/{project_id}/images/{image_id}/suggestions/segment/box",
    response_model=SuggestionSessionResponse,
    summary="Segment from bounding box"
)
async def segment_from_box(
    project_id: UUID,
    image_id: int,
    request: SAMBoxRequest,
    ctx: ProjectContext = Depends(get_project_context),
    seg_svc: SegmentationService = Depends(get_segmentation_service),
    sug_svc: SuggestionService = Depends(get_suggestion_service)
):
    """User draws box → returns refined mask."""
    session = await sug_svc.get_session(request.session_id)
    
    image = await sug_svc.load_image(image_id)
    box = (request.box.x, request.box.y, request.box.width, request.box.height)
    output = seg_svc.segment_from_box(
        image, 
        box,
        model=session.model_name,
        device=session.model_device,
        precision=session.model_precision
    )
    
    suggestion = Suggestion(
        suggestion_id=str(uuid_lib.uuid4()),
        bbox=BoundingBox(x=output.bbox[0], y=output.bbox[1],
                         width=output.bbox[2], height=output.bbox[3]),
        mask_rle=output.mask_rle,
        polygons=output.polygons,
        confidence=output.confidence,
        type='box',
    )
    
    await sug_svc.store_suggestions(request.session_id, [suggestion])
    await sug_svc.update_session_source_type(request.session_id, SuggestionSourceType.SAM_BOX)
    
    return SuggestionSessionResponse(
        session_id=request.session_id,
        status="ready",
        config=ModelConfig(
            model=session.model_name,
            device=session.model_device,
            precision=session.model_precision    
        ),
        source_type=SuggestionSourceType.SAM_BOX,
        suggestions=[suggestion],
        count=len([suggestion]) 
    )
