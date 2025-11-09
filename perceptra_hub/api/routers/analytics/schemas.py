from typing import (
    Optional,
    Dict,
    List,
    Any,
)
from pydantic import Field, BaseModel

# ============= Response Models =============

class ProjectSummaryResponse(BaseModel):
    """Project overview statistics."""
    project_id: str
    project_name: str
    description: Optional[str]
    project_type: str
    visibility: str
    created_at: str
    last_edited: str
    
    # Image stats
    total_images: int
    annotated_images: int
    reviewed_images: int
    finalized_images: int
    null_images: int
    
    # Annotation stats
    total_annotations: int
    manual_annotations: int
    prediction_annotations: int
    
    # Job stats
    total_jobs: int
    active_jobs: int
    completed_jobs: int
    
    # Version stats
    total_versions: int
    latest_version: Optional[str]


class ImageStatsResponse(BaseModel):
    """Detailed image statistics."""
    total: int
    by_status: Dict[str, int]
    by_split: Dict[str, int]
    by_job_status: Dict[str, int]
    upload_trend: List[Dict[str, Any]]
    average_dimensions: Dict[str, float]
    average_file_size_mb: float


class AnnotationStatsResponse(BaseModel):
    """Annotation statistics."""
    total: int
    active: int
    inactive: int
    by_source: Dict[str, int]
    by_status: Dict[str, int]
    average_per_image: float
    class_distribution: List[Dict[str, Any]]


class ClassDistribution(BaseModel):
    """Class distribution item."""
    class_id: int
    class_name: str
    color: str
    count: int
    percentage: float


class AnnotationGroupResponse(BaseModel):
    """Annotation group with classes."""
    id: int
    name: str
    description: Optional[str]
    classes: List[ClassDistribution]
    total_annotations: int


class JobStatsResponse(BaseModel):
    """Job statistics."""
    total: int
    by_status: Dict[str, int]
    total_images: int
    average_images_per_job: float
    completion_rate: float


class VersionStatsResponse(BaseModel):
    """Version statistics."""
    id: str
    version_name: str
    version_number: int
    export_format: str
    export_status: str
    total_images: int
    by_split: Dict[str, int]
    total_annotations: int
    file_size_mb: Optional[float]
    created_at: str
    exported_at: Optional[str]


class EvaluationStatsResponse(BaseModel):
    """Model evaluation statistics."""
    total_evaluated: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    by_class: List[Dict[str, Any]]