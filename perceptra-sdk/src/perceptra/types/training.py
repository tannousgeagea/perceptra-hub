from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict

class TrainingSession(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[float] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    metrics: Optional[dict] = None
    best_metrics: Optional[dict] = None
    config: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model_name: Optional[str] = None
    project_name: Optional[str] = None
