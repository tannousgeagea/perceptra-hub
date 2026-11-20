"""
Active Learning API Endpoints
"""

from collections import defaultdict
from fastapi import APIRouter, Depends, Query
from typing import Dict, List, Optional
from pydantic import BaseModel

from projects.models import Project, ProjectImage
from annotations.models import Annotation, AnnotationAudit
from django.db.models import Count, Min, Max, Avg, Q
from api.routers.active_learning.schemas import *
from api.dependencies import ProjectContext, get_project_context

router = APIRouter(prefix="/active-learning",)


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get(
    "/projects/{project_id}/suggest",
    response_model=List[PriorityImage],
    summary="Get prioritized images for review"
)
def suggest_images_to_review(
    strategy: str = Query("hybrid", description="uncertainty|diversity|error_prone|hybrid"),
    limit: int = Query(50, ge=1, le=200),
    project_ctx: ProjectContext = Depends(get_project_context),
):
    """
    {project_id} is the project's UUID; the caller needs access to the project.
    """
    project_ctx.require_project_access()
    return _prioritize(project_ctx.project, strategy, limit)


# Unreviewed images considered per request (newest first).
POOL_SIZE = 2000
REVIEWED_STATUSES = ("reviewed", "approved", "dataset")
# A class's past miss rate counts once it has this many audited predictions.
MIN_AUDITS = 5
WEIGHTS = {"uncertainty": 0.4, "rarity": 0.25, "error_prone": 0.25, "density": 0.1}


def _class_rarity(project: Project) -> Dict[str, float]:
    """1 for the rarest class among reviewed annotations, 0 for the most common."""
    counts = dict(
        Annotation.objects.filter(
            project_image__project=project, project_image__status__in=REVIEWED_STATUSES,
            is_active=True, is_deleted=False,
        ).values_list("annotation_class__name").annotate(n=Count("id")).values_list("annotation_class__name", "n")
    )
    most = max(counts.values(), default=0)
    return {name: 1 - n / most for name, n in counts.items()} if most else {}


def _class_error_rate(project: Project) -> Dict[str, float]:
    """Per class: share of audited predictions that were wrong (false positive or missed)."""
    rows = (
        AnnotationAudit.objects.filter(annotation__project_image__project=project)
        .values("annotation__annotation_class__name")
        .annotate(
            tp=Count("id", filter=Q(evaluation_status="TP")),
            fp=Count("id", filter=Q(evaluation_status="FP")),
            fn=Count("id", filter=Q(evaluation_status="FN")),
        )
    )
    rates = {}
    for r in rows:
        total = r["tp"] + r["fp"] + r["fn"]
        if total >= MIN_AUDITS:
            rates[r["annotation__annotation_class__name"]] = (r["fp"] + r["fn"]) / total
    return rates


def _prioritize(project: Project, strategy: str, limit: int) -> List[PriorityImage]:
    """
    Unreviewed images with model predictions, ranked by how much a human look
    would help. Signals (each 0-1):
    - uncertainty: 1 - the lowest prediction confidence
    - rarity ("diversity"): its predicted classes are rare among reviewed annotations
    - error_prone: its predicted classes were often wrong in past reviews
    - density: many objects
    hybrid weighs them 0.4 / 0.25 / 0.25 / 0.1.
    """
    strategy = strategy.replace("-", "_")
    pool_ids = list(
        ProjectImage.objects.filter(project=project, is_active=True, reviewed=False, marked_as_null=False)
        .exclude(status__in=REVIEWED_STATUSES)
        .filter(annotations__annotation_source="prediction", annotations__is_active=True,
                annotations__is_deleted=False)
        .order_by("-added_at").values_list("id", flat=True).distinct()[:POOL_SIZE]
    )
    if not pool_ids:
        return []

    predictions: Dict[int, list] = defaultdict(list)
    for pi_id, confidence, cls in Annotation.objects.filter(
        project_image_id__in=pool_ids, annotation_source="prediction", is_active=True, is_deleted=False,
    ).values_list("project_image_id", "confidence", "annotation_class__name"):
        predictions[pi_id].append((confidence, cls))

    rarity_of = _class_rarity(project)
    error_of = _class_error_rate(project)

    scored = []
    for pi_id, preds in predictions.items():
        confidences = [c for c, _ in preds if c is not None]
        classes = sorted({cls for _, cls in preds if cls})
        lowest = min(confidences) if confidences else None
        scores = {
            "uncertainty": 1 - lowest if lowest is not None else 0.0,
            # A class never reviewed yet is as rare as it gets.
            "rarity": max((rarity_of.get(c, 1.0) for c in classes), default=0.0),
            "error_prone": max((error_of.get(c, 0.0) for c in classes), default=0.0),
            "density": min(len(preds) / 20, 1.0),
        }
        if strategy == "hybrid":
            priority = sum(scores[k] * w for k, w in WEIGHTS.items())
        else:
            priority = scores.get({"diversity": "rarity"}.get(strategy, strategy), scores["uncertainty"])
        scored.append((priority, pi_id, preds, classes, lowest, scores))

    scored.sort(key=lambda s: s[0], reverse=True)
    top = scored[:limit]
    images = {
        pi.id: pi for pi in ProjectImage.objects.filter(id__in=[s[1] for s in top])
        .select_related("image", "image__storage_profile")
    }
    # Already queued for someone: shown, but not offered for another job.
    from jobs.models import JobImage
    open_job_of = dict(
        JobImage.objects.filter(
            project_image_id__in=images.keys(), is_active=True,
            job__is_active=True, job__status__in=("unassigned", "assigned", "in_review"),
        ).values_list("project_image_id", "job__name")
    )

    result = []
    for rank, (priority, pi_id, preds, classes, lowest, scores) in enumerate(top, start=1):
        pi = images[pi_id]
        reasons = []
        if lowest is not None and lowest < 0.5:
            reasons.append(f"Lowest confidence {lowest:.2f}")
        rare = [c for c in classes if rarity_of.get(c, 1.0) >= 0.8]
        if rare:
            reasons.append(f"Rare class: {', '.join(rare[:3])}")
        risky = sorted((c for c in classes if error_of.get(c, 0) >= 0.3), key=lambda c: -error_of[c])
        if risky:
            reasons.append(f"'{risky[0]}' was wrong {error_of[risky[0]]:.0%} of the time")
        if len(preds) >= 15:
            reasons.append(f"{len(preds)} objects")
        try:
            thumbnail = pi.image.get_download_url(expiration=3600)
        except Exception:  # unreachable storage: the list still works without a picture
            thumbnail = None
        result.append(PriorityImage(
            image_id=str(pi.image.image_id),
            image_name=pi.image.name,
            project_image_id=pi.id,
            thumbnail_url=thumbnail,
            priority_score=round(priority, 4),
            priority_rank=rank,
            reasons=reasons or ["Unreviewed model predictions"],
            num_predictions=len(preds),
            lowest_confidence=lowest,
            classes=classes,
            scores=PriorityScores(**{k: round(v, 4) for k, v in scores.items()}),
            in_open_job=open_job_of.get(pi_id),
        ))
    return result


@router.get(
    "/projects/{project_id}/batch",
    response_model=BatchSuggestion,
    summary="Get balanced batch for review"
)
def suggest_review_batch(
    batch_size: int = Query(20, ge=1, le=100),
    project_ctx: ProjectContext = Depends(get_project_context),
):
    """
    Get a balanced batch of images ensuring class diversity.
    """
    
    # Get suggestions
    project_ctx.require_project_access()
    suggestions = _prioritize(project_ctx.project, "hybrid", batch_size * 2)
    
    # Balance by class (simple implementation)
    selected = []
    class_counts = {}
    
    for img in suggestions:
        if len(selected) >= batch_size:
            break
        
        # Check class balance
        img_classes = set(img.classes)
        overrepresented = any(
            class_counts.get(cls, 0) >= batch_size / max(len(class_counts), 1) + 2
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