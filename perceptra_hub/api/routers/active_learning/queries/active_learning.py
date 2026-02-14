"""
Active Learning API Endpoints
"""

from fastapi import APIRouter, Query, Path
from typing import List, Optional
from pydantic import BaseModel

from projects.models import Project, ProjectImage
from annotations.models import Annotation
from django.db.models import Count, Min, Max, Avg
from api.routers.active_learning.schemas import *

router = APIRouter(prefix="/active-learning",)


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get(
    "/projects/{project_id}/suggest",
    response_model=List[PriorityImage],
    summary="Get prioritized images for review"
)
async def suggest_images_to_review(
    project_id: int = Path(...),
    strategy: str = Query("hybrid", description="uncertainty|diversity|error_prone|hybrid"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Get images prioritized for review using active learning.
    
    Strategies:
    - uncertainty: Low confidence predictions
    - diversity: Under-represented classes
    - error_prone: Similar to known error patterns
    - hybrid: Balanced combination (recommended)
    """
    
    # Get unreviewed images with predictions
    images = ProjectImage.objects.filter(
        project_id=project_id,
        reviewed=False,
        annotated=True,
        is_active=True
    ).prefetch_related('annotations', 'image')
    
    prioritized = []
    
    for idx, project_img in enumerate(images[:limit * 2]):  # Get more for filtering
        
        # Get predictions
        predictions = project_img.annotations.filter(
            annotation_source='prediction',
            is_active=True
        )
        
        if not predictions.exists():
            continue
        
        # Calculate scores
        confidences = [p.confidence for p in predictions if p.confidence]
        lowest_conf = min(confidences) if confidences else None
        
        # Uncertainty score (lower confidence = higher priority)
        uncertainty_score = 1 - lowest_conf if lowest_conf else 0.5
        
        # Diversity score (placeholder - would need full implementation)
        diversity_score = 0.5
        
        # Error likelihood score (placeholder)
        error_score = 0.7 if predictions.count() > 10 else 0.3
        
        # Complexity score
        complexity_score = min(predictions.count() / 20, 1.0)
        
        # Hybrid strategy
        if strategy == "hybrid":
            priority_score = (
                uncertainty_score * 0.4 +
                diversity_score * 0.2 +
                error_score * 0.3 +
                complexity_score * 0.1
            )
        elif strategy == "uncertainty":
            priority_score = uncertainty_score
        elif strategy == "diversity":
            priority_score = diversity_score
        else:
            priority_score = error_score
        
        # Reasons
        reasons = []
        if lowest_conf and lowest_conf < 0.5:
            reasons.append("Low confidence predictions")
        if predictions.count() > 15:
            reasons.append("High object density")
        if lowest_conf and lowest_conf > 0.9:
            reasons.append("High confidence (verify quality)")
        
        # Get classes
        classes = list(predictions.values_list('annotation_class__name', flat=True).distinct())
        
        prioritized.append(PriorityImage(
            image_id=str(project_img.image.image_id),
            image_name=project_img.image.name,
            priority_score=priority_score,
            priority_rank=0,  # Will be set after sorting
            reasons=reasons or ["General review needed"],
            num_predictions=predictions.count(),
            lowest_confidence=lowest_conf,
            classes=classes
        ))
    
    # Sort by priority
    prioritized.sort(key=lambda x: x.priority_score, reverse=True)
    
    # Assign ranks
    for rank, img in enumerate(prioritized[:limit], start=1):
        img.priority_rank = rank
    
    return prioritized[:limit]


@router.get(
    "/projects/{project_id}/batch",
    response_model=BatchSuggestion,
    summary="Get balanced batch for review"
)
async def suggest_review_batch(
    project_id: int = Path(...),
    batch_size: int = Query(20, ge=1, le=100),
):
    """
    Get a balanced batch of images ensuring class diversity.
    """
    
    # Get suggestions
    suggestions = await suggest_images_to_review(
        project_id=project_id,
        strategy="hybrid",
        limit=batch_size * 2
    )
    
    # Balance by class (simple implementation)
    selected = []
    class_counts = {}
    
    for img in suggestions:
        if len(selected) >= batch_size:
            break
        
        # Check class balance
        img_classes = set(img.classes)
        overrepresented = any(
            class_counts.get(cls, 0) >= batch_size / len(class_counts) + 2
            for cls in img_classes
        )
        
        if not overrepresented or len(selected) < batch_size / 2:
            selected.append(img)
            for cls in img_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
    
    # Summary
    summary = {
        "avg_priority": sum(img.priority_score for img in selected) / len(selected) if selected else 0,
        "class_distribution": class_counts,
        "total_predictions": sum(img.num_predictions for img in selected),
    }
    
    return BatchSuggestion(
        images=selected,
        total_suggested=len(selected),
        strategy_used="hybrid_balanced",
        summary=summary
    )