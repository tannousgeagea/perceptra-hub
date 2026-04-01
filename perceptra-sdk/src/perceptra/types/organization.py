from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, ConfigDict

class OrganizationMember(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int = 0
    username: str = ""
    email: str = ""
    first_name: str = ""
    last_name: str = ""
    role: Optional[str] = None
    status: Optional[str] = None
    joined_at: Optional[str] = None

class OrganizationDetails(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[int] = None
    org_id: Optional[str] = None
    name: str = ""
    slug: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    logo_url: Optional[str] = None
    subscription_plan: Optional[str] = None
    status: Optional[str] = None
    statistics: Optional[dict] = None
    recent_members: List[OrganizationMember] = []
