from typing import List, Optional
from datetime import datetime, timedelta, date
from pydantic import BaseModel

class UserActivitySummary(BaseModel):
    user_id: str
    username: str
    full_name: str
    
    # Annotation metrics
    total_annotations: int
    manual_annotations: int
    ai_predictions_edited: int
    ai_predictions_accepted: int
    
    # Review metrics
    images_reviewed: int
    images_finalized: int
    
    # Quality metrics
    avg_annotation_time_seconds: Optional[float]
    avg_edit_magnitude: Optional[float]
    
    # Activity
    last_activity: Optional[datetime]
    total_sessions: int


class ProjectProgressSummary(BaseModel):
    project_id: str
    project_name: str
    
    # Progress breakdown
    total_images: int
    images_unannotated: int
    images_annotated: int
    images_reviewed: int
    images_finalized: int
    completion_percentage: float
    
    # Quality insights
    untouched_predictions: int
    edited_predictions: int
    rejected_predictions: int
    prediction_acceptance_rate: float
    
    # Velocity
    annotations_per_hour: Optional[float]
    active_contributors: int


class ActivityTimelineEvent(BaseModel):
    event_id: str
    event_type: str
    user: str
    timestamp: datetime
    project: Optional[str]
    metadata: dict


class LeaderboardEntry(BaseModel):
    rank: int
    user_id: str
    username: str
    full_name: str
    metric_value: int
    percentage_of_total: float


class PredictionQualityMetrics(BaseModel):
    total_predictions: int
    untouched: int
    accepted_without_edit: int
    minor_edits: int
    major_edits: int
    class_changes: int
    rejected: int
    
    # Percentages
    untouched_percentage: float
    acceptance_rate: float
    avg_edit_magnitude: Optional[float]

class OrgActivitySummaryResponse(BaseModel):
    date: date
    total_events: int
    active_users: int
    annotation_events: int
    image_events: int
    annotations_created: int
    images_reviewed: int
    images_uploaded: int

class OrganizationActivitySummary(BaseModel):
    organization_id: str
    organization_name: str
    period: dict
    
    # Overall metrics
    # Annotation metrics
    total_annotations: int
    manual_annotations: int
    ai_predictions_edited: int
    ai_predictions_accepted: int
    
    # Review metrics
    images_reviewed: int
    images_finalized: int
    
    # Quality metrics
    avg_annotation_time_seconds: Optional[float]
    avg_edit_magnitude: Optional[float]
    
    
    # User metrics
    total_active_users: int
    avg_annotations_per_user: float
    
    # Project metrics
    total_projects: int
    active_projects: int
    
    # Top performers
    top_annotator: Optional[dict]
    top_reviewer: Optional[dict]
    
class ActivityTrendPoint(BaseModel):
    date: str
    annotations: int
    reviews: int
    uploads: int
    active_users: int
