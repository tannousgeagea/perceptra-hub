# routes/models.py
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import RequestContext, get_request_context, require_permission
from typing import List
from pydantic import BaseModel
from ml_models.models import ModelTag


router = APIRouter(
    responses={404: {"description": "Not found"}},
)

class TagOut(BaseModel):
    id: int
    name: str
    description: Optional[str]

    class Config:
        from_attributes = True

class TagIn(BaseModel):
    name: str
    description: Optional[str] = None

@router.get("/tags", response_model=List[TagOut])
def list_tags(ctx: RequestContext = Depends(get_request_context)):
    return ModelTag.objects.filter(organization=ctx.organization).order_by("name")

@router.post("/tags", response_model=TagOut)
def create_tag(data: TagIn, ctx: RequestContext = Depends(require_permission("write"))):
    """Create a model tag in the organization (org editor and up)."""
    if ModelTag.objects.filter(organization=ctx.organization, name=data.name).exists():
        raise HTTPException(status_code=400, detail="Tag already exists")

    tag = ModelTag.objects.create(
        organization=ctx.organization, name=data.name, description=data.description
    )
    return tag
