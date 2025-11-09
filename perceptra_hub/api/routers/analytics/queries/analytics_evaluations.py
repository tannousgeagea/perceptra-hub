"""
Production-ready analytics and statistics endpoints.

Features:
- Optimized queries with select_related/prefetch_related
- Caching with Redis
- Async support
- Proper error handling
- Response models with validation
"""
from fastapi import APIRouter, Depends
from uuid import UUID
import logging
from asgiref.sync import sync_to_async

from django.db.models import Count, Q

from api.dependencies import get_project_context, ProjectContext
from projects.models import Project
from annotations.models import AnnotationAudit
from api.routers.analytics.schemas import EvaluationStatsResponse
from api.routers.analytics.utils import cache_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Analytics"])

# ============= Evaluation Statistics =============

@router.get(
    "/{project_id}/analytics/evaluation",
    response_model=EvaluationStatsResponse,
    summary="Get Evaluation Statistics"
)
@cache_response(timeout=300)
async def get_evaluation_stats(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context)
):
    """Get model evaluation metrics (TP/FP/FN)."""
    
    @sync_to_async
    def get_stats(project: Project):
        audits_qs = AnnotationAudit.objects.filter(
            annotation__project_image__project=project
        )
        
        # Overall metrics
        stats = audits_qs.aggregate(
            tp=Count('id', filter=Q(evaluation_status='TP')),
            fp=Count('id', filter=Q(evaluation_status='FP')),
            fn=Count('id', filter=Q(evaluation_status='FN'))
        )
        
        tp = stats['tp'] or 0
        fp = stats['fp'] or 0
        fn = stats['fn'] or 0
        total = tp + fp + fn
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Per-class metrics
        class_stats = audits_qs.values(
            'annotation__annotation_class__name'
        ).annotate(
            tp=Count('id', filter=Q(evaluation_status='TP')),
            fp=Count('id', filter=Q(evaluation_status='FP')),
            fn=Count('id', filter=Q(evaluation_status='FN'))
        )
        
        by_class = []
        for item in class_stats:
            cls_tp = item['tp']
            cls_fp = item['fp']
            cls_fn = item['fn']
            
            cls_precision = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0
            cls_recall = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
            
            by_class.append({
                "class_name": item['annotation__annotation_class__name'],
                "tp": cls_tp,
                "fp": cls_fp,
                "fn": cls_fn,
                "precision": round(cls_precision, 3),
                "recall": round(cls_recall, 3),
                "f1_score": round(cls_f1, 3)
            })
        
        return {
            "total_evaluated": total,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "by_class": by_class
        }
    
    return await get_stats(project_ctx.project)
