"""
Production-Grade Model Evaluation API Endpoint
==============================================

FastAPI endpoint for retrieving evaluated annotations with complete audit trail.
Designed for high-performance, caching, and future extensibility.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, model_validator

# ============================================================================
# RESPONSE MODELS (Pydantic Schemas)
# ============================================================================

class EvaluationStatus(str, Enum):
    TP = "TP"
    FP = "FP"
    FN = "FN"
    PENDING = "pending"


class AnnotationSource(str, Enum):
    MANUAL = "manual"
    PREDICTION = "prediction"


class EditType(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"
    CLASS_CHANGE = "class_change"
    DELETED = "deleted"


class OriginalPrediction(BaseModel):
    """Original model prediction before human edits"""
    bbox: List[float] = Field(..., min_items=4, max_items=4)
    class_id: int
    class_name: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    class Config:
        frozen = True  # Immutable


class AnnotationEvaluation(BaseModel):
    """Evaluation metadata from AnnotationAudit"""
    status: Optional[EvaluationStatus]
    was_edited: bool = False
    edit_magnitude: Optional[float] = Field(None, ge=0.0, le=1.0)
    edit_type: Optional[EditType]
    localization_iou: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Match metadata (for TPs)
    matched_annotation_id: Optional[str] = None
    match_iou: Optional[float] = None
    
    # Review metadata
    reviewed_by: Optional[str]
    reviewed_at: Optional[datetime]


class AnnotationResponse(BaseModel):
    """Complete annotation with evaluation data"""
    
    # Identity
    id: str
    annotation_uid: str
    
    # Current state (post-review)
    class_id: int
    class_name: str
    bbox: List[float] = Field(..., min_items=4, max_items=4, description="[x1, y1, x2, y2] normalized 0-1")
    
    # Source & provenance
    source: AnnotationSource
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    model_version: Optional[str] = None
    
    created_by: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    # Review state
    reviewed: bool
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    
    # Original prediction (if edited)
    original_prediction: Optional[OriginalPrediction] = None
    
    # Evaluation
    evaluation: Optional[AnnotationEvaluation] = None
    
    # Status flags
    is_active: bool
    is_deleted: bool
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class ImageEvaluationSummary(BaseModel):
    """Pre-computed image-level metrics"""
    
    # Counts
    tp: int = 0
    fp: int = 0
    fn: int = 0
    pending: int = 0
    
    # Breakdown
    tp_unedited: int = 0  # Perfect predictions
    tp_minor_edit: int = 0
    tp_major_edit: int = 0
    tp_class_change: int = 0
    
    # Quality indicators
    total_predictions: int = 0
    total_ground_truth: int = 0  # TP + FN
    
    # Statistics (for non-zero samples)
    mean_confidence: Optional[float] = None
    mean_tp_confidence: Optional[float] = None
    mean_fp_confidence: Optional[float] = None
    
    mean_edit_magnitude: Optional[float] = None  # For edited TPs
    mean_localization_iou: Optional[float] = None
    
    # Flags for UI filtering
    has_errors: bool = False  # Has FP or FN
    has_edits: bool = False
    is_fully_reviewed: bool = False
    
    @model_validator(mode="after")
    def compute_flags(self):
        self.has_errors = self.fp > 0 or self.fn > 0
        self.has_edits = (
            self.tp_minor_edit +
            self.tp_major_edit +
            self.tp_class_change
        ) > 0
        return self
    


class ImageResponse(BaseModel):
    """Image with annotations and evaluation summary"""
    
    # Image metadata
    image_id: UUID
    name: str
    display_name: str
    
    width: Optional[int]
    height: Optional[int]
    megapixels: Optional[float]
    
    # Status
    status: Optional[str]
    annotated: Optional[bool]
    reviewed: Optional[bool]
    
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    
    # Tags for filtering
    tags: List[str] = []
    
    # Annotations with evaluation
    annotations: List[AnnotationResponse]
    
    # Pre-computed summary
    evaluation: Optional[ImageEvaluationSummary] = None
    
    class Config:
        from_attributes = True


class DatasetEvaluationSummary(BaseModel):
    """Dataset-level aggregated metrics"""
    
    # Dataset info
    total_images: int = 0
    reviewed_images: int = 0
    unreviewed_images: int = 0
    
    # Annotation counts
    total_annotations: int = 0
    total_predictions: int = 0
    total_manual: int = 0
    
    # Evaluation counts
    tp: int = 0
    fp: int = 0
    fn: int = 0
    pending: int = 0
    
    # Derived metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Quality metrics
    edit_rate: float = 0.0  # % of TPs edited
    hallucination_rate: float = 0.0  # FP / predictions
    miss_rate: float = 0.0  # FN / ground_truth
    
    # Per-class breakdown available via separate endpoint
    class_count: int = 0


class EvaluationResponse(BaseModel):
    """Complete evaluation API response"""
    
    # Metadata
    requested_at: datetime = Field(default_factory=datetime.now)
    filter_applied: Dict[str, Any] = {}
    
    # Data
    images: List[ImageResponse] = []
    
    # Summary
    summary: DatasetEvaluationSummary
    
    # Pagination
    total_count: int = 0
    page: int = 1
    page_size: int = 100
    has_next: bool = False
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class PerClassMetrics(BaseModel):
    """Per-class performance breakdown"""
    
    class_id: int
    class_name: str
    
    tp: int = 0
    fp: int = 0
    fn: int = 0
    
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Sample counts
    total_predictions: int = 0
    total_ground_truth: int = 0
    
    # Quality indicators
    edit_rate: Optional[float] = None
    mean_confidence: Optional[float] = None
    
# ============================================================================
# HELPER FOR SCHEMA VALIDATION
# ============================================================================

def validate_schemas():
    """
    Test function to ensure all schemas can be instantiated with minimal data.
    Run this during testing to catch validation issues early.
    """
    
    # Test DatasetEvaluationSummary with empty data
    summary = DatasetEvaluationSummary()
    assert summary.total_images == 0
    assert summary.tp == 0
    assert summary.precision == 0.0
    print("✓ DatasetEvaluationSummary validates with defaults")
    
    # Test ImageEvaluationSummary
    img_summary = ImageEvaluationSummary()
    assert img_summary.tp == 0
    assert img_summary.has_errors == False
    print("✓ ImageEvaluationSummary validates with defaults")
    
    # Test EvaluationResponse with minimal data
    response = EvaluationResponse(summary=summary)
    assert response.total_count == 0
    assert len(response.images) == 0
    print("✓ EvaluationResponse validates with minimal data")
    
    print("\n✅ All schemas validated successfully!")


if __name__ == "__main__":
    validate_schemas()