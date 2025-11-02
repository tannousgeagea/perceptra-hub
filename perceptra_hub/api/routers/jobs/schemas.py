from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class AssigneeOut(BaseModel):
    id: str
    username: str
    email: str


class JobResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    image_count: int
    assignee: Optional[AssigneeOut]
    batch_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    
class JobUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[str] = Field(
        None,
        description="Job status: unassigned, assigned, in_review, completed, sliced"
    )
    assignee_id: Optional[str] = Field(
        None,
        description="User ID to assign job to (empty string to unassign)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Batch 1 - Vehicles",
                "description": "Annotate all vehicle images",
                "status": "in_review",
                "assignee_id": "user-uuid-here"
            }
        }

class JobSplitRequest(BaseModel):
    number_of_slices: int = Field(..., ge=2, description="Number of slices to split job into")
    user_assignments: List[Optional[str]] = Field(
        ...,
        description="List of user IDs for each slice (null for unassigned)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "number_of_slices": 3,
                "user_assignments": ["user-uuid-1", "user-uuid-2", None]
            }
        }
