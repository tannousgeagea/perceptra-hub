from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, ConfigDict

class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    status: Optional[str] = None
    metrics: Optional[dict] = None
    created_at: Optional[str] = None
