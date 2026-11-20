"""
Active Learning API Endpoints
"""

from fastapi import APIRouter, Query, Path
from typing import List, Optional
from pydantic import BaseModel

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PriorityScores(BaseModel):
    """Each signal 0-1 (higher = more worth a human look)."""
    uncertainty: float   # 1 - lowest prediction confidence
    rarity: float        # how rare its predicted classes are among reviewed annotations
    error_prone: float   # highest past miss rate (FP+FN) of its predicted classes
    density: float       # objects in the image, 20+ = 1


class PriorityImage(BaseModel):
    """Prioritized image for review"""
    image_id: str
    image_name: str
    # ProjectImage id: what /projects/{id}/jobs/from-images takes.
    project_image_id: int
    thumbnail_url: Optional[str] = None
    priority_score: float
    priority_rank: int
    reasons: List[str]
    num_predictions: int
    lowest_confidence: Optional[float]
    classes: List[str]
    scores: Optional[PriorityScores] = None
    # Name of the open job it's already in (it can't go into another one), if any.
    in_open_job: Optional[str] = None


class BatchSuggestion(BaseModel):
    """Batch of suggested images"""
    images: List[PriorityImage]
    total_suggested: int
    strategy_used: str
    summary: dict