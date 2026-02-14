
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class AlertResponse(BaseModel):
    """Alert response model"""
    id: int
    project_id: int
    project_name: str
    severity: str
    metric_name: str
    current_value: float
    previous_value: Optional[float]
    threshold_value: float
    change_percent: Optional[float]
    message: str
    created_at: datetime
    is_acknowledged: bool
    acknowledged_at: Optional[datetime]
    acknowledged_by: Optional[str]
    
    class Config:
        from_attributes = True


class AlertStats(BaseModel):
    """Alert statistics"""
    total_alerts: int
    critical_count: int
    warning_count: int
    unacknowledged_count: int
    acknowledged_count: int


class AcknowledgeRequest(BaseModel):
    """Request to acknowledge alert"""
    acknowledged_by: str
