# api/routers/suggestions/schemas.py

from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from enum import Enum
from uuid import UUID

class SuggestionSourceType(str, Enum):
    SAM_AUTO = "sam_auto"
    SAM_POINT = "sam_point"
    SAM_BOX = "sam_box"
    SAM_TEXT = "sam_text"           # SAM3
    SIMILAR_OBJECT = "similar"       # SAM3 exemplar
    PREVIOUS_FRAME = "prev_frame"
    LABEL_SUGGEST = "label"

class BoundingBox(BaseModel):
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)
    width: float = Field(..., ge=0, le=1)
    height: float = Field(..., ge=0, le=1)

class PointPrompt(BaseModel):
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)
    label: int = Field(1, description="1=foreground, 0=background")

class Suggestion(BaseModel):
    suggestion_id: str
    id: str = ""  # mirrors suggestion_id; populated by validator so frontend gets a stable string key
    bbox: BoundingBox
    mask_rle: Optional[dict] = None
    polygons: Optional[List] = None
    confidence: float
    suggested_class_id: Optional[int] = None
    suggested_class_name: Optional[str] = None
    type: str = "point"
    status: str = "pending"
    suggested_label: Optional[str] = Field(None, alias="suggested_class_name")

    @model_validator(mode='after')
    def _sync_id(self) -> 'Suggestion':
        if not self.id:
            self.id = self.suggestion_id
        return self
    

class ModelConfig(BaseModel):
    """Model configuration - set once per session."""
    model: str = Field("sam_v2", pattern="^(sam_v1|sam_v2|sam_v3)$")
    device: str = Field("cuda", pattern="^(cuda|cpu)$")
    precision: str = Field("fp16", pattern="^(fp16|fp32)$")

class SuggestionSessionResponse(BaseModel):
    session_id: UUID
    status: str  # pending, ready, failed
    source_type: Optional[SuggestionSourceType] = None
    config: ModelConfig  # Return active config
    suggestions: List[Suggestion] = []
    count: int = 0  # Add this

class SessionCreateRequest(BaseModel):
    """Initialize AI assistance session with model selection."""
    config: ModelConfig = ModelConfig()
    
# === Request Models ===

class SAMAutoRequest(BaseModel):
    """Auto-segment entire image."""
    session_id: UUID
    points_per_side: int = Field(32, ge=8, le=64)
    pred_iou_thresh: float = Field(0.88, ge=0.0, le=1.0)
    stability_score_thresh: float = Field(0.95, ge=0.0, le=1.0)
    min_area: float = Field(0.001, description="Min bbox area as fraction of image")

class SAMPointRequest(BaseModel):
    """Segment from point prompts."""
    points: List[PointPrompt]
    session_id: UUID

class SAMBoxRequest(BaseModel):
    box: BoundingBox
    session_id: UUID

class SAMTextRequest(BaseModel):
    """SAM3 text prompt."""
    text: str = Field(..., min_length=1, max_length=500)
    session_id: UUID
    
class SAMExemplarRequest(BaseModel):
    """SAM3 exemplar-based similarity search."""
    session_id: UUID
    reference_annotation_uid: Optional[str] = None
    box: Optional[BoundingBox] = None  # Alternative to annotation reference

class SimilarObjectRequest(BaseModel):
    """Find similar objects to a reference annotation."""
    reference_annotation_uid: str
    max_suggestions: int = Field(10, ge=1, le=50)
    min_similarity: float = Field(0.7, ge=0, le=1)

class PreviousFrameRequest(BaseModel):
    """Propagate annotations from previous image."""
    source_image_id: int
    annotation_uids: Optional[List[str]] = None  # None = all
    session_id: Optional[UUID] = None  # active SAM session to append into (optional)

class SegmentationResponse(BaseModel):
    """Lightweight response for non-SAM suggestion endpoints."""
    suggestions: List[Suggestion] = []
    count: int = 0

class LabelSuggestionRequest(BaseModel):
    """Suggest labels for a bbox region."""
    bbox: BoundingBox
    top_k: int = Field(5, ge=1, le=20)

class AcceptSuggestionsRequest(BaseModel):
    suggestion_ids: List[str]
    class_id: Optional[str] = None
    class_name: Optional[str] = None
    use_polygon: bool = Field(
        True,
        description="Store the SAM polygon contour in addition to the bounding box. "
                    "Set False to create a plain bounding-box annotation.",
    )
    
class RejectSuggestionsRequest(BaseModel):
    suggestion_ids: List[str]