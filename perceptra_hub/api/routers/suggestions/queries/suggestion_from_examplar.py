
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
    "/{project_id}/images/{image_id}/suggestions/segment/exemplar",
    response_model=SuggestionSessionResponse,
    summary="Find similar objects to exemplar (SAM3)"
)
async def segment_from_exemplar(
    project_id: UUID,
    image_id: int,
    request: SAMExemplarRequest,
    ctx: ProjectContext = Depends(get_project_context),
    seg_svc: SegmentationService = Depends(get_segmentation_service),
    sug_svc: SuggestionService = Depends(get_suggestion_service)
):
    """
    User labels one object → finds all similar.
    Maps directly to your Feature #2 (Similar Object Suggestions).
    """
    session = await sug_svc.get_session(request.session_id)
    
    image = await sug_svc.load_image(image_id)
    
    # Get exemplar box from existing annotation or request
    if request.reference_annotation_uid:
        ann = await sug_svc.get_annotation(request.reference_annotation_uid)
        box = (ann.data[0], ann.data[1], 
               ann.data[2] - ann.data[0], ann.data[3] - ann.data[1])
        suggested_class_id = ann.annotation_class.class_id
        suggested_class_name = ann.annotation_class.name
    else:
        box = (request.box.x, request.box.y, request.box.width, request.box.height)
        suggested_class_id = None
        suggested_class_name = None
    
    outputs = seg_svc.segment_from_exemplar(
        image, 
        box,
        model=session.model_name,
        device=session.model_device,
        precision=session.model_precision 
    )
    
    suggestions = [
        Suggestion(
            id=str(uuid_lib.uuid4()),
            bbox=BoundingBox(x=o.bbox[0], y=o.bbox[1],
                             width=o.bbox[2], height=o.bbox[3]),
            mask_rle=o.mask_rle,
            polygons=o.polygons,
            confidence=o.confidence,
            suggested_class_id=suggested_class_id,
            suggested_class_name=suggested_class_name,
            type="similar"
        )
        for o in outputs
    ]
    
    await sug_svc.store_suggestions(request.session_id, suggestions)
    await sug_svc.update_session_source_type(request.session_id, SuggestionSourceType.SIMILAR_OBJECT)
    
    return SuggestionSessionResponse(
        session_id=request.session_id,
        status="ready",
        source_type=SuggestionSourceType.SIMILAR_OBJECT,
        suggestions=suggestions,
        config=ModelConfig(...),
        count=len(suggestions)
    )
