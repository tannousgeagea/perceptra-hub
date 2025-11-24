
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
    "/{project_id}/images/{image_id}/suggestions/segment/points",
    response_model=SuggestionSessionResponse,
    summary="Segment from point clicks"
)
async def segment_from_points(
    project_id: UUID,
    image_id: int,
    request: SAMPointRequest,
    ctx: ProjectContext = Depends(get_project_context),
    seg_svc: SegmentationService = Depends(get_segmentation_service),
    sug_svc: SuggestionService = Depends(get_suggestion_service)
):
    """
    User clicks positive/negative points → returns single mask.
    Fast enough for real-time interaction.
    """
    session = await sug_svc.get_session(request.session_id)
    
    # Load image
    image = await sug_svc.load_image(image_id)
    
    # Run segmentation
    points = [(p.x, p.y, p.label) for p in request.points]
    output = seg_svc.segment_from_points(
        image, 
        points,
        model=session.model_name,
        device=session.model_device,
        precision=session.model_precision
        
    )
    
    suggestion = Suggestion(
        id=str(uuid_lib.uuid4()),
        bbox=BoundingBox(x=output.bbox[0], y=output.bbox[1], 
                         width=output.bbox[2], height=output.bbox[3]),
        mask_rle=output.mask_rle,
        polygons=output.polygons,
        confidence=output.confidence,
        type="point",
        status="pending",
    )
    
    await sug_svc.store_suggestions(request.session_id, [suggestion])
    await sug_svc.update_session_source_type(request.session_id, SuggestionSourceType.SAM_POINT)
    
    return SuggestionSessionResponse(
        session_id=request.session_id,
        status="ready",
        source_type=SuggestionSourceType.SAM_POINT,
        suggestions=[suggestion],
        config=ModelConfig(...),
        count=len([suggestion])
    )
