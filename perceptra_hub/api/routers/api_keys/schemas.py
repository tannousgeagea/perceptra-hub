"""
FastAPI routes for API Key management.
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
import logging


# ============= Pydantic Schemas =============

class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Friendly name for the API key")
    description: Optional[str] = Field(None, description="Optional description")
    scope: str = Field("organization", pattern="^(organization|user)$")
    user_id: Optional[int] = Field(None, description="User ID if scope=user")
    permission: str = Field("read", pattern="^(read|write|admin)$")
    expires_in_days: int = Field(90, ge=1, le=365, description="Days until expiration (1-365)")
    rate_limit_per_minute: int = Field(60, ge=0, description="Requests per minute (0=unlimited)")
    rate_limit_per_hour: int = Field(1000, ge=0, description="Requests per hour (0=unlimited)")


class APIKeyResponse(BaseModel):
    id: int
    prefix: str
    name: str
    description: Optional[str]
    scope: str
    user_id: Optional[int]
    user_username: Optional[str]
    permission: str
    is_active: bool
    usage_count: int
    last_used_at: Optional[datetime]
    created_at: datetime
    expires_at: datetime
    rate_limit_per_minute: int
    rate_limit_per_hour: int
    created_by_username: str


class APIKeyCreateResponse(BaseModel):
    message: str
    api_key: str  # Full key - only shown once!
    key_info: APIKeyResponse


class APIKeyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    permission: Optional[str] = Field(None, pattern="^(read|write|admin)$")
    is_active: Optional[bool] = None
    rate_limit_per_minute: Optional[int] = Field(None, ge=0)
    rate_limit_per_hour: Optional[int] = Field(None, ge=0)