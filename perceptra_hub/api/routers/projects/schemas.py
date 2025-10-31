

from typing import Optional, List
from uuid import UUID
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
