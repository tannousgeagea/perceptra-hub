from __future__ import annotations
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class ProjectStatistics(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_images: int = 0
    total_annotations: int = 0
    annotation_groups: int = 0

class Project(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    project_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    project_type: Optional[dict] = None
    visibility: Optional[dict] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_edited: Optional[str] = None

class ProjectListItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    project_id: str
    name: str
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    project_type_name: str = ""
    visibility_name: str = ""
    is_active: bool = True
    statistics: Optional[ProjectStatistics] = None
    created_at: Optional[str] = None
    last_edited: Optional[str] = None
    user_role: Optional[str] = None

class SplitDatasetResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    message: str = ""
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    total_split: int = 0
    already_split: int = 0
