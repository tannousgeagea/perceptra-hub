from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, ConfigDict

class JobProgress(BaseModel):
    model_config = ConfigDict(extra="allow")
    total: int = 0
    annotated: int = 0
    reviewed: int = 0
    completed: int = 0

class Job(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    name: str = ""
    description: Optional[str] = None
    status: Optional[str] = None
    image_count: int = 0
    assignee: Optional[dict] = None
    batch_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    progress: Optional[JobProgress] = None
