from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, ConfigDict

class StorageProfile(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    name: str = ""
    backend: Optional[str] = None
    region: Optional[str] = None
    is_default: bool = False
    is_active: bool = True
    config: Optional[dict] = None
    created_at: Optional[str] = None
