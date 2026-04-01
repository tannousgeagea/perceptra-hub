from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict

class Version(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    version_id: Optional[str] = None
    version_name: Optional[str] = None
    description: Optional[str] = None
    format: Optional[str] = None
    status: Optional[str] = None
    item_count: Optional[int] = None
    class_distribution: Optional[dict] = None
    download_count: int = 0
    created_at: Optional[str] = None
    created_by: Optional[str] = None
