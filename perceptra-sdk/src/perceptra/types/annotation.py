from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, ConfigDict

class Annotation(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    annotation_uid: Optional[str] = None
    type: Optional[str] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    color: Optional[str] = None
    data: List[float] = []
    source: Optional[str] = None
    confidence: Optional[float] = None
    reviewed: bool = False
    is_active: bool = True
    created_at: Optional[str] = None
    created_by: Optional[str] = None

class AnnotationCreateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    message: Optional[str] = None
    annotation_id: Optional[str] = None
    annotation_uid: Optional[str] = None
    created: Optional[bool] = None
