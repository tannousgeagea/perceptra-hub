"""
Temporal Analysis API Schemas
"""

from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class TrendPoint(BaseModel):
    """Single point in time series"""
    date: datetime
    precision: float
    recall: float
    f1_score: float
    edit_rate: float
    hallucination_rate: float
    tp: int
    fp: int
    fn: int


class SnapshotResponse(BaseModel):
    """Snapshot summary"""
    id: int
    snapshot_date: datetime
    model_version: Optional[str]
    precision: float
    recall: float
    f1_score: float
    edit_rate: float
    total_images: int
    
    class Config:
        from_attributes = True


class TrendAnalysis(BaseModel):
    """Trend analysis response"""
    metric_name: str
    current_value: float
    previous_value: Optional[float]
    change_percent: Optional[float]
    direction: str  # 'improving', 'stable', 'degrading'
    mean_7d: Optional[float]
    mean_30d: Optional[float]


class TemporalResponse(BaseModel):
    """Complete temporal analysis"""
    project_id: int
    project_name: str
    date_range: tuple[datetime, datetime]
    trends: List[TrendPoint]
    analysis: List[TrendAnalysis]
    snapshot_count: int