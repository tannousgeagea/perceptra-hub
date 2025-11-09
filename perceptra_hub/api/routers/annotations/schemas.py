from pydantic import BaseModel, Field
from typing import Optional, List

class AnnotationCreate(BaseModel):
    annotation_type: str
    annotation_id: Optional[int] = None
    annotation_class_id: Optional[int] = None
    annotation_class_name: Optional[str] = None
    data: List[float] = Field(..., description="[xmin, ymin, xmax, ymax] in normalized coords")
    annotation_uid: Optional[str] = None
    annotation_source: str = Field(default="manual", pattern="^(manual|prediction)$")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "annotation_type_id": 1,
                "annotation_class_id": 5,
                "data": [0.1, 0.2, 0.5, 0.8],
                "annotation_source": "manual"
            }
        }


class AnnotationUpdate(BaseModel):
    annotation_class_id: Optional[int] = None
    data: Optional[List[float]] = None
    reviewed: Optional[bool] = None
    is_active: Optional[bool] = None


class AnnotationBatchCreate(BaseModel):
    project_image_id: int
    annotations: List[AnnotationCreate]

class AnnotationResponse(BaseModel):
    id: str
    annotation_uid: Optional[str]
    type: str
    class_id: int
    class_name: str
    color: Optional[str]
    data: List[float]
    source: str
    confidence: Optional[float]
    reviewed: bool
    is_active: bool
    created_at: str
    created_by: Optional[str]
    
class AnnotationCreateResponse(BaseModel):
    message: str
    annotation: AnnotationResponse
    
class AnnotationAuditConfig:
    MINOR_AUDIT_THRESHOLD:float = 0.3
    MAJOR_AUDIT_THRESHOLD:float = 0.7