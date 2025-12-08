
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from datetime import datetime


# ============= Pydantic Schemas =============

class ModelCreateRequest(BaseModel):
    """Request schema for creating a new model"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    task: str = Field(..., description="Task type (object-detection, classification, etc.)")
    framework: str = Field(..., description="Framework (yolo, pytorch, tensorflow, etc.)")
    tags: List[str] = Field(default_factory=list)


class ModelArtifactResponse(BaseModel):
    """Artifact URLs for a model version"""
    checkpoint: Optional[str] = None
    onnx: Optional[str] = None
    logs: Optional[str] = None


class DatasetInfoResponse(BaseModel):
    """Dataset information for a model version"""
    id: str
    name: str
    version: str
    item_count: int
    created_at: datetime


class ModelVersionResponse(BaseModel):
    """Response schema for model version"""
    id: str
    version_number: int
    version_name: Optional[str] = None
    status: str
    deployment_status: str
    metrics: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    dataset: Optional[DatasetInfoResponse] = None
    artifacts: ModelArtifactResponse
    created_by: Optional[str] = None
    created_at: datetime
    deployed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ModelDetailResponse(BaseModel):
    """Detailed model information with versions"""
    id: str
    name: str
    description: str
    task: str
    framework: str
    tags: List[str]
    project_id: str
    project_name: str
    versions: List[ModelVersionResponse]
    latest_version: Optional[ModelVersionResponse] = None
    production_version: Optional[ModelVersionResponse] = None
    created_by: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ModelListResponse(BaseModel):
    """List item for models"""
    id: str
    name: str
    description: str
    task: str
    framework: str
    tags: List[str]
    version_count: int
    latest_version_number: Optional[int] = None
    latest_status: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TrainingTriggerRequest(BaseModel):
    """Request to trigger model training"""
    dataset_version_id: str
    parent_version_id: Optional[str] = Field(
        None,
        description="Base version ID for transfer learning"
    )
    config: dict = Field(
        default_factory=dict,
        description="Training configuration (hyperparameters, etc.)"
    )
    version_name: Optional[str] = Field(None, max_length=255)
    compute_profile_id: Optional[str] = Field(
        None,
        description="Compute profile to use (uses default if not specified)"
    )

class TrainingTriggerResponse(BaseModel):
    """Response after triggering training"""
    model_version_id: str
    version_number: int
    training_session_id: str
    task_id: str
    status: str
    message: str