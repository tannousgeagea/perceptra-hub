from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict

class ModelVersion(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    version_number: Optional[int] = None
    version_name: Optional[str] = None
    status: Optional[str] = None
    deployment_status: Optional[str] = None
    metrics: Optional[dict] = None
    config: Optional[dict] = None
    created_at: Optional[str] = None

class Model(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    model_id: Optional[str] = None
    name: str = ""
    description: Optional[str] = None
    task: Optional[str] = None
    framework: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class ModelDetail(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    model_id: Optional[str] = None
    name: str = ""
    description: Optional[str] = None
    task: Optional[str] = None
    framework: Optional[str] = None
    versions: List[ModelVersion] = []
    latest_version: Optional[ModelVersion] = None
    production_version: Optional[ModelVersion] = None

class TrainingTriggerResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_version_id: Optional[str] = None
    version_number: Optional[int] = None
    training_session_id: Optional[str] = None
    task_id: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None
