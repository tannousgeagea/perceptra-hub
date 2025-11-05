

from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


# ============= Schemas =============

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    project_type_id: int
    visibility_id: Optional[int] = None
    thumbnail_url: Optional[str] = None
    settings: dict = Field(default_factory=dict)


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    project_type_id: Optional[int] = None
    visibility_id: Optional[int] = None
    thumbnail_url: Optional[str] = None
    is_active: Optional[bool] = None
    settings: Optional[dict] = None


class UserBasicInfo(BaseModel):
    id: int
    username: str
    email: str
    first_name: str
    last_name: str

class ProjectStatistics(BaseModel):
    total_images: int
    total_annotations: int = 0
    annotation_groups: int

class ProjectListItem(BaseModel):
    id: int
    project_id: str
    name: str
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    project_type_name: str
    visibility_name: str
    is_active: bool
    statistics: ProjectStatistics
    created_by: Optional[UserBasicInfo] = None
    updated_by: Optional[UserBasicInfo] = None
    created_at: str
    last_edited: str
    user_role: str = Field(..., description="Current user's role in this project")

class ProjectResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: Optional[str]
    project_type: dict
    visibility: dict
    is_active: bool
    is_deleted: bool
    created_at: str
    updated_at: str
    last_edited: str


class AddImagesToProjectRequest(BaseModel):
    image_ids: List[str] = Field(..., min_items=1)
    mode_id: Optional[int] = None
    priority: int = Field(default=0)
    auto_assign_job: bool = Field(
        default=True,
        description="Automatically assign to available job"
    )


class JobCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    assignee_id: Optional[str] = None
    batch_id: Optional[str] = None


class JobUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    assignee_id: Optional[str] = None
    status: Optional[str] = None


class AssignedUserOut(BaseModel):
    id: int
    username: str
    email: str
    avatar: Optional[str] = None

class JobProgress(BaseModel):
    total: int
    annotated: int
    reviewed: int
    completed: int

class JobResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    image_count: int
    assignee: Optional[dict]
    assignedUser: Optional[AssignedUserOut] = None
    batch_id: Optional[str]
    created_at: str
    updated_at: str
    progress: Optional[JobProgress] = None


class AssignImagesToJobRequest(BaseModel):
    project_image_ids: List[str] = Field(..., min_items=1)


class StorageProfileOut(BaseModel):
    id: str
    name: str
    backend: str

class ImageDetail(BaseModel):
    id: str
    image_id: str
    name: str
    original_filename: Optional[str]
    width: Optional[int]
    height: Optional[int]
    aspect_ratio: Optional[float]
    file_format: Optional[str]
    file_size: Optional[int]
    file_size_mb: Optional[float]
    megapixels: Optional[float]
    storage_key: Optional[str]
    checksum: Optional[str]
    source_of_origin: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    uploaded_by: Optional[str]
    tags: List[str] = []
    storage_profile: Optional[StorageProfileOut]
    download_url: Optional[str]
    status: Optional[str] = None
    annotated: Optional[bool] = None
    reviewed: Optional[bool] = None
    marked_as_null:Optional[bool] = None
    priority: Optional[int] = 0
    job_assignment_status: Optional[str] = None
    added_at: Optional[datetime] = None 

class JobSummary(BaseModel):
    id: str
    name: str
    status: str
    assignee: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class AnnotationOut(BaseModel):
    id: str
    annotation_uid: str
    type: Optional[str]
    class_id: int
    class_name: str
    color: str
    data: List[float]
    source: Optional[str]
    confidence: Optional[float]
    reviewed: bool
    is_active: bool
    created_at: str
    created_by: Optional[str]


class ProjectImageOut(BaseModel):
    id: str
    image: ImageDetail
    status: Optional[str]
    annotated: bool
    reviewed: bool
    finalized: bool
    marked_as_null: bool
    priority: Optional[int]
    job_assignment_status: Optional[str]
    mode: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    added_by: Optional[str]
    reviewed_by: Optional[str]
    added_at: str
    reviewed_at: Optional[str]
    updated_at: str
    jobs: List[JobSummary]
    annotations: List[AnnotationOut]
    
class ProjectImage(ImageDetail):
    annotations: List[AnnotationOut]
    
class ProjectImagesResponse(BaseModel):
    total: int
    annotated: int
    unannotated: int
    reviewed: int
    images:List[ProjectImage]
    
class JobImagesResponce(ProjectImagesResponse):
    job: JobSummary
    
   
class ProjectImageStatusUpdate(BaseModel):
    status: Optional[str] = Field(None, pattern="^(unannotated|in_progress|annotated|reviewed|approved|rejected|dataset)$")
    reviewed: Optional[bool] = None
    marked_as_null: Optional[bool] = None
    finalized: Optional[bool] = None
    feedback: Optional[str] = None
    
class SplitDatasetRequest(BaseModel):
    train_ratio: float = Field(..., ge=0.0, le=1.0, description="Train split ratio (0.0-1.0)")
    val_ratio: float = Field(..., ge=0.0, le=1.0, description="Validation split ratio (0.0-1.0)")
    test_ratio: float = Field(..., ge=0.0, le=1.0, description="Test split ratio (0.0-1.0)")
    
    def validate_ratios(self):
        """Validate that ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "train_ratio": 0.7,
                "val_ratio": 0.2,
                "test_ratio": 0.1
            }
        }


class SplitDatasetResponse(BaseModel):
    message: str
    train_count: int
    val_count: int
    test_count: int
    total_split: int
    already_split: int