from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, ConfigDict

class ComputeProfile(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    name: str = ""
    provider: Optional[str] = None
    instance_type: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None
