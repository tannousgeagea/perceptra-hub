
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from datetime import datetime

# ============= Schemas =============

class ProviderInstanceInfo(BaseModel):
    """Instance type information"""
    name: str
    vcpus: int
    memory_gb: float
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    cost_per_hour: Optional[float] = None


class ComputeProviderResponse(BaseModel):
    """Available compute provider"""
    id: int
    name: str
    provider_type: str
    description: str
    requires_user_credentials: bool
    is_active: bool
    available_instances: List[Dict[str, Any]]
    
    class Config:
        from_attributes = True


class ComputeProfileCreateRequest(BaseModel):
    """Request to create compute profile"""
    name: str = Field(..., min_length=1, max_length=255)
    provider_id: int = Field(..., description="Primary compute provider ID")
    default_instance_type: str = Field(..., description="Default instance to use")
    strategy: str = Field(
        default="queue",
        description="Strategy: cheapest, fastest, preferred, queue"
    )
    max_concurrent_jobs: int = Field(default=5, ge=1, le=50)
    max_cost_per_hour: Optional[float] = Field(None, ge=0)
    max_training_hours: int = Field(default=24, ge=1, le=168)
    user_credentials: Optional[Dict[str, Any]] = Field(
        None,
        description="Provider credentials (encrypted at rest)"
    )
    is_default: bool = False


class ComputeProfileUpdateRequest(BaseModel):
    """Request to update compute profile"""
    name: Optional[str] = None
    default_instance_type: Optional[str] = None
    strategy: Optional[str] = None
    max_concurrent_jobs: Optional[int] = Field(None, ge=1, le=50)
    max_cost_per_hour: Optional[float] = Field(None, ge=0)
    max_training_hours: Optional[int] = Field(None, ge=1, le=168)
    user_credentials: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    is_default: Optional[bool] = None


class ComputeProfileResponse(BaseModel):
    """Compute profile response"""
    id: str
    name: str
    provider: ComputeProviderResponse
    default_instance_type: str
    strategy: str
    max_concurrent_jobs: int
    max_cost_per_hour: Optional[float]
    max_training_hours: int
    has_credentials: bool
    is_active: bool
    is_default: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class FallbackProviderRequest(BaseModel):
    """Add fallback provider"""
    provider_id: int
    priority: int = Field(..., ge=1, description="Priority (1=highest)")