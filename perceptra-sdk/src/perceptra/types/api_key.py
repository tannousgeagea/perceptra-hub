from __future__ import annotations
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class APIKey(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[int] = None
    api_key_id: Optional[str] = None
    key_prefix: Optional[str] = None
    name: str = ""
    description: Optional[str] = None
    scope: Optional[str] = None
    permissions: Optional[str] = None
    is_active: bool = True
    usage_count: int = 0
    last_used_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    scopes: List[str] = []
    allowed_ips: List[str] = []
    version: int = 1
    created_by_username: Optional[str] = None

class APIKeyCreateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    message: str = ""
    api_key: str = ""
    key_info: Optional[APIKey] = None
