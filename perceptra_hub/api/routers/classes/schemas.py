from pydantic import BaseModel
from typing import Optional


class AnnotationClassOut(BaseModel):
    id: int
    classId: int
    name: str
    color: str
    count: int

class AnnotationClassCreate(BaseModel):
    name: str
    color: str
    count: Optional[int] = 0
    description: Optional[str] = None
    
class AnnotationClassUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None