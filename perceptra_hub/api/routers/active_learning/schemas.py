"""
Active Learning API Endpoints
"""

from fastapi import APIRouter, Query, Path
from typing import List, Optional
from pydantic import BaseModel

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PriorityImage(BaseModel):
    """Prioritized image for review"""
    image_id: str
    image_name: str
    priority_score: float
    priority_rank: int
    reasons: List[str]
    num_predictions: int
    lowest_confidence: Optional[float]
    classes: List[str]


class BatchSuggestion(BaseModel):
    """Batch of suggested images"""
    images: List[PriorityImage]
    total_suggested: int
    strategy_used: str
    summary: dict