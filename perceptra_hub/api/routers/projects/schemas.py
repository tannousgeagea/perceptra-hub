

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

class JobSummary(BaseModel):
    id: str
    name: str
    status: str


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