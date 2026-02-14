"""
Production-Grade Model Evaluation API Endpoint - Django ORM Implementation
===========================================================================

FastAPI endpoint using Django ORM for retrieving evaluated annotations.
Optimized queries with select_related, prefetch_related for N+1 prevention.
"""

import orjson
import time
from fastapi.routing import APIRoute
from fastapi import Request, Response
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from uuid import UUID
from enum import Enum

from django.db.models import (
    Prefetch, Q, Count, Avg, Case, When, F, Value,
    CharField, FloatField, IntegerField, OuterRef, Subquery
)
from django.db.models.functions import Coalesce
from django.core.paginator import Paginator

from fastapi import APIRouter, Query, HTTPException, Path
from fastapi.responses import StreamingResponse
from asgiref.sync import sync_to_async

from annotations.models import Annotation, AnnotationAudit, ProjectImage, AnnotationClass
from projects.models import Project
from api.routers.evaluation.schemas import *
from api.routers.evaluation.cache import get_evaluation_cache, CacheKeyBuilder


# ============================================================================
# OPTIMIZED DJANGO ORM QUERIES
# ============================================================================

class EvaluationQueryBuilder:
    """
    Builds optimized Django ORM queries for evaluation data.
    Uses select_related, prefetch_related to prevent N+1 queries.
    """
    
    @staticmethod
    @sync_to_async
    def get_evaluated_images(
        project_id: int,
        model_version: Optional[str] = None,
        reviewed_only: bool = False,
        has_errors_only: bool = False,
        class_ids: Optional[List[int]] = None,
        min_confidence: Optional[float] = None,
        evaluation_status: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> tuple[List[ProjectImage], int]:
        """
        Fetch images with annotations and audits using optimized Django ORM.
        
        Returns:
            Tuple of (images list, total count)
        """
        
        # Base query with optimized prefetching
        query = ProjectImage.objects.filter(
            project_id=project_id,
            is_active=True
        ).select_related(
            'project'
        ).prefetch_related(
            # Prefetch annotations with their related data
            Prefetch(
                'annotations',
                queryset=Annotation.objects.filter(
                    is_deleted=False,
                    is_active=True
                ).select_related(
                    'annotation_class',
                    'annotation_type',
                    'created_by',
                    'updated_by',
                    'reviewed_by',
                    'original_class'  # For edited predictions
                ).prefetch_related(
                    # Prefetch audit data
                    Prefetch(
                        'audit',
                        queryset=AnnotationAudit.objects.select_related('reviewed_by')
                    )
                )
            )
        )
        
        # Apply filters
        if reviewed_only:
            query = query.filter(reviewed=True)
        
        if model_version:
            # Filter images that have annotations from this model version
            query = query.filter(
                annotations__model_version=model_version,
                annotations__is_deleted=False
            ).distinct()
        
        if has_errors_only:
            # Images with FP or FN
            query = query.filter(
                Q(annotations__audit__evaluation_status='FP') |
                Q(annotations__audit__evaluation_status='FN')
            ).distinct()
        
        if class_ids:
            query = query.filter(
                annotations__annotation_class_id__in=class_ids
            ).distinct()
        
        if min_confidence:
            query = query.filter(
                annotations__confidence__gte=min_confidence
            ).distinct()
        
        if evaluation_status:
            query = query.filter(
                annotations__audit__evaluation_status__in=evaluation_status
            ).distinct()
        
        # Get total count before pagination
        total_count = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        images = list(query[offset:offset + page_size])
        
        return images, total_count
    
    @staticmethod
    def compute_image_summary(image: ProjectImage) -> ImageEvaluationSummary:
        """Compute evaluation summary for a single image"""
        
        summary = ImageEvaluationSummary()
        
        tp_confidences = []
        fp_confidences = []
        all_confidences = []
        edit_magnitudes = []
        localization_ious = []
        
        for ann in image.annotations.all():
            # Get audit if exists
            try:
                audit = ann.audit
            except AnnotationAudit.DoesNotExist:
                audit = None
            
            # Count by evaluation status
            if audit and audit.evaluation_status:
                status = audit.evaluation_status
                
                if status == 'TP':
                    summary.tp += 1
                    
                    # Breakdown by edit type
                    if not audit.was_edited:
                        summary.tp_unedited += 1
                    elif audit.edit_type == 'minor':
                        summary.tp_minor_edit += 1
                    elif audit.edit_type == 'major':
                        summary.tp_major_edit += 1
                    elif audit.edit_type == 'class_change':
                        summary.tp_class_change += 1
                    
                    if ann.confidence:
                        tp_confidences.append(ann.confidence)
                    
                    if audit.edit_magnitude:
                        edit_magnitudes.append(audit.edit_magnitude)
                    
                    if audit.localization_iou:
                        localization_ious.append(audit.localization_iou)
                
                elif status == 'FP':
                    summary.fp += 1
                    if ann.confidence:
                        fp_confidences.append(ann.confidence)
                
                elif status == 'FN':
                    summary.fn += 1
            else:
                summary.pending += 1
            
            # Count predictions vs manual
            if ann.annotation_source == 'prediction':
                summary.total_predictions += 1
                if ann.confidence:
                    all_confidences.append(ann.confidence)
        
        # Ground truth = TP + FN
        summary.total_ground_truth = summary.tp + summary.fn
        
        # Compute means
        if all_confidences:
            summary.mean_confidence = sum(all_confidences) / len(all_confidences)
        if tp_confidences:
            summary.mean_tp_confidence = sum(tp_confidences) / len(tp_confidences)
        if fp_confidences:
            summary.mean_fp_confidence = sum(fp_confidences) / len(fp_confidences)
        if edit_magnitudes:
            summary.mean_edit_magnitude = sum(edit_magnitudes) / len(edit_magnitudes)
        if localization_ious:
            summary.mean_localization_iou = sum(localization_ious) / len(localization_ious)
        
        # Flags
        summary.is_fully_reviewed = (summary.pending == 0)
        
        return summary
    
    @staticmethod
    def compute_dataset_summary(image_summaries: List[ImageEvaluationSummary]) -> DatasetEvaluationSummary:
        """Aggregate image summaries to dataset level"""
        
        summary = DatasetEvaluationSummary(
            total_images=len(image_summaries),
            reviewed_images=sum(1 for s in image_summaries if s.is_fully_reviewed),
            unreviewed_images=sum(1 for s in image_summaries if not s.is_fully_reviewed),
            total_annotations=0,
            total_predictions=0,
            total_manual=0,
            tp=0,
            fp=0,
            fn=0,
            pending=0,
        )
        
        # Aggregate counts
        for img_summary in image_summaries:
            summary.tp += img_summary.tp
            summary.fp += img_summary.fp
            summary.fn += img_summary.fn
            summary.pending += img_summary.pending
            summary.total_predictions += img_summary.total_predictions
            summary.total_manual += img_summary.fn  # FNs are manual
        
        summary.total_annotations = summary.tp + summary.fp + summary.fn + summary.pending
        
        # Compute derived metrics
        if summary.tp + summary.fp > 0:
            summary.precision = summary.tp / (summary.tp + summary.fp)
        
        if summary.tp + summary.fn > 0:
            summary.recall = summary.tp / (summary.tp + summary.fn)
        
        if summary.precision + summary.recall > 0:
            summary.f1_score = 2 * (summary.precision * summary.recall) / (summary.precision + summary.recall)
        
        # Quality metrics
        if summary.tp > 0:
            edited_tps = sum(
                s.tp_minor_edit + s.tp_major_edit + s.tp_class_change 
                for s in image_summaries
            )
            summary.edit_rate = edited_tps / summary.tp
        
        if summary.total_predictions > 0:
            summary.hallucination_rate = summary.fp / summary.total_predictions
        
        if summary.tp + summary.fn > 0:
            summary.miss_rate = summary.fn / (summary.tp + summary.fn)
        
        return summary
    
    @staticmethod
    @sync_to_async
    def get_class_metrics(
        project_id: int,
        model_version: Optional[str] = None,
        reviewed_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get per-class performance metrics using optimized aggregation.
        
        Uses Django ORM aggregation for performance.
        """
        
        # Build annotation queryset
        annotations = Annotation.objects.filter(
            project_image__project_id=project_id,
            is_deleted=False,
            is_active=True
        ).select_related('annotation_class', 'audit')
        
        if reviewed_only:
            annotations = annotations.filter(project_image__reviewed=True)
        
        if model_version:
            annotations = annotations.filter(model_version=model_version)
        
        # Aggregate by class
        class_stats = annotations.values(
            'annotation_class_id',
            'annotation_class__name'
        ).annotate(
            tp=Count(Case(When(audit__evaluation_status='TP', then=1))),
            fp=Count(Case(When(audit__evaluation_status='FP', then=1))),
            fn=Count(Case(When(audit__evaluation_status='FN', then=1))),
            total_predictions=Count(Case(
                When(annotation_source='prediction', then=1)
            )),
            mean_confidence=Avg(
                Case(
                    When(confidence__isnull=False, then=F('confidence')),
                    output_field=FloatField()
                )
            ),
            edit_rate=Avg(
                Case(
                    When(
                        audit__evaluation_status='TP',
                        audit__was_edited=True,
                        then=Value(1.0)
                    ),
                    default=Value(0.0),
                    output_field=FloatField()
                )
            )
        ).order_by('-tp')
        
        # Compute derived metrics
        results = []
        for stat in class_stats:
            tp = stat['tp']
            fp = stat['fp']
            fn = stat['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results.append({
                'class_id': stat['annotation_class_id'],
                'class_name': stat['annotation_class__name'],
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1_score, 4),
                'total_predictions': stat['total_predictions'],
                'total_ground_truth': tp + fn,
                'mean_confidence': round(stat['mean_confidence'], 3) if stat['mean_confidence'] else None,
                'edit_rate': round(stat['edit_rate'], 3) if stat['edit_rate'] else None,
            })
        
        return results
    
    @staticmethod
    @sync_to_async
    def get_quick_summary(
        project_id: int,
        model_version: Optional[str] = None
    ) -> DatasetEvaluationSummary:
        """
        Optimized summary using database aggregation only.
        Much faster than fetching all images.
        """
        
        # Get image counts
        images = ProjectImage.objects.filter(
            project_id=project_id,
            is_active=True
        )
        
        total_images = images.count()
        reviewed_images = images.filter(reviewed=True).count()
        
        # Get annotation stats
        annotations = Annotation.objects.filter(
            project_image__project_id=project_id,
            is_deleted=False,
            is_active=True
        )
        
        if model_version:
            annotations = annotations.filter(model_version=model_version)
        
        # Aggregate metrics
        stats = annotations.aggregate(
            total_annotations=Count('id'),
            total_predictions=Count(Case(
                When(annotation_source='prediction', then=1)
            )),
            total_manual=Count(Case(
                When(annotation_source='manual', then=1)
            )),
            tp=Count(Case(When(audit__evaluation_status='TP', then=1))),
            fp=Count(Case(When(audit__evaluation_status='FP', then=1))),
            fn=Count(Case(When(audit__evaluation_status='FN', then=1))),
            pending=Count(Case(When(audit__evaluation_status__isnull=True, then=1)))
        )
        
        # Build summary
        summary = DatasetEvaluationSummary(
            total_images=total_images,
            reviewed_images=reviewed_images,
            unreviewed_images=total_images - reviewed_images,
            total_annotations=stats['total_annotations'],
            total_predictions=stats['total_predictions'],
            total_manual=stats['total_manual'],
            tp=stats['tp'],
            fp=stats['fp'],
            fn=stats['fn'],
            pending=stats['pending'],
        )
        
        # Compute derived metrics
        if summary.tp + summary.fp > 0:
            summary.precision = summary.tp / (summary.tp + summary.fp)
        
        if summary.tp + summary.fn > 0:
            summary.recall = summary.tp / (summary.tp + summary.fn)
        
        if summary.precision + summary.recall > 0:
            summary.f1_score = 2 * (summary.precision * summary.recall) / (summary.precision + summary.recall)
        
        if summary.total_predictions > 0:
            summary.hallucination_rate = summary.fp / summary.total_predictions
        
        if summary.tp + summary.fn > 0:
            summary.miss_rate = summary.fn / (summary.tp + summary.fn)
        
        return summary


# ============================================================================
# TRANSFORMER FUNCTIONS
# ============================================================================

def transform_annotation_to_response(ann: Annotation) -> AnnotationResponse:
    """Transform Django Annotation model to Pydantic response"""
    
    # Get audit safely
    try:
        audit = ann.audit
    except AnnotationAudit.DoesNotExist:
        audit = None
    
    # Build base response
    response = AnnotationResponse(
        id=str(ann.id),
        annotation_uid=ann.annotation_uid,
        class_id=ann.annotation_class.id,
        class_name=ann.annotation_class.name,
        bbox=ann.data,
        source=AnnotationSource(ann.annotation_source),
        confidence=ann.confidence,
        model_version=ann.model_version,
        created_by=ann.created_by.username if ann.created_by else None,
        created_at=ann.created_at,
        updated_at=ann.updated_at,
        reviewed=ann.reviewed,
        reviewed_by=ann.reviewed_by.username if ann.reviewed_by else None,
        reviewed_at=ann.reviewed_at,
        is_active=ann.is_active,
        is_deleted=ann.is_deleted,
    )
    
    # Add original prediction if exists
    if ann.original_data:
        response.original_prediction = OriginalPrediction(
            bbox=ann.original_data,
            class_id=ann.original_class.id if ann.original_class else ann.annotation_class.id,
            class_name=ann.original_class.name if ann.original_class else ann.annotation_class.name,
            confidence=ann.confidence,
        )
    
    # Add evaluation data from audit
    if audit:
        response.evaluation = AnnotationEvaluation(
            status=EvaluationStatus(audit.evaluation_status) if audit.evaluation_status else None,
            was_edited=audit.was_edited,
            edit_magnitude=audit.edit_magnitude,
            edit_type=EditType(audit.edit_type) if audit.edit_type else None,
            localization_iou=audit.localization_iou,
            matched_annotation_id=str(audit.matched_manual_annotation.id) if audit.matched_manual_annotation else None,
            match_iou=audit.iou,
            reviewed_by=audit.reviewed_by.username if audit.reviewed_by else None,
            reviewed_at=audit.reviewed_at,
        )
    
    return response

@sync_to_async
def transform_image_to_response(
    img: ProjectImage,
    include_summary: bool = True
) -> ImageResponse:
    """Transform Django ProjectImage to Pydantic response"""
    
    # Transform annotations
    annotations = [
        transform_annotation_to_response(ann)
        for ann in img.annotations.all()
    ]
    
    # Compute summary if requested
    evaluation = None
    if include_summary:
        query_builder = EvaluationQueryBuilder()
        evaluation = query_builder.compute_image_summary(img)
    
    # Build response
    response = ImageResponse(
        image_id=img.image.image_id,
        name=img.image.name,
        display_name=img.image.original_filename or img.image.name,
        width=img.image.width,
        height=img.image.height,
        megapixels=img.image.megapixels,
        status=img.status,
        annotated=img.annotated,
        reviewed=img.reviewed,
        created_at=img.added_at,
        updated_at=img.updated_at,
        tags=[tag.name for tag in img.image.tags.all()],
        annotations=annotations,
        evaluation=evaluation,
    )
    
    return response


# ============================================================================
# API ROUTER
# ============================================================================

class TimedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            print(f"route duration: {duration}")
            print(f"route response: {response}")
            print(f"route response headers: {response.headers}")
            return response

        return custom_route_handler

router = APIRouter(prefix="/evaluation", route_class=TimedRoute)


@router.get(
    "/projects/{project_id}/images",
    response_model=EvaluationResponse,
    summary="Get evaluated images with annotations"
)
async def get_project_evaluation(
    project_id: int = Path(..., description="Project ID"),
    
    # Filtering
    model_version: Optional[str] = Query(None, description="Filter by model version"),
    reviewed_only: bool = Query(False, description="Only reviewed images"),
    has_errors_only: bool = Query(False, description="Only images with FP/FN"),
    class_ids: Optional[List[int]] = Query(None, description="Filter by class IDs"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    evaluation_status: Optional[List[EvaluationStatus]] = Query(None),
    
    # Pagination
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    """
    Retrieve images with annotations and evaluation metrics.
    
    Optimized with Django ORM prefetching for high performance.
    """
    
    # Fetch images using optimized query
    cache = get_evaluation_cache()
    
    # Build filters for cache key
    filters = {
        "model_version": model_version,
        "reviewed_only": reviewed_only,
        "has_errors_only": has_errors_only,
        "class_ids": class_ids,
        "min_confidence": min_confidence,
        "evaluation_status": [s.value for s in evaluation_status] if evaluation_status else None,
    }
    
    # Build cache key
    cache_key = CacheKeyBuilder.image_list_key(project_id, page, page_size, filters)
    
    async def compute():
        query_builder = EvaluationQueryBuilder()
        try:
            images_data, total_count = await query_builder.get_evaluated_images(
                project_id=project_id,
                model_version=model_version,
                reviewed_only=reviewed_only,
                has_errors_only=has_errors_only,
                class_ids=class_ids,
                min_confidence=min_confidence,
                evaluation_status=[s.value for s in evaluation_status] if evaluation_status else None,
                page=page,
                page_size=page_size,
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
        
        # Transform to response models
        images = []
        image_summaries = []
        
        for img_data in images_data:
            try:
                img_response = await transform_image_to_response(img_data, include_summary=True)
                images.append(img_response)
                
                if img_response.evaluation:
                    image_summaries.append(img_response.evaluation)
            except Exception as e:
                # Log but don't fail entire request for one bad image
                print(f"Error transforming image {img_data.image_id}: {e}")
                continue
        
        # Compute dataset summary
        dataset_summary = query_builder.compute_dataset_summary(image_summaries)
        
        # Build response
        response = EvaluationResponse(
            images=images,
            summary=dataset_summary,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=page * page_size < total_count,
            filter_applied={
                "model_version": model_version,
                "reviewed_only": reviewed_only,
                "has_errors_only": has_errors_only,
                "class_ids": class_ids,
                "min_confidence": min_confidence,
            }
        )
        
        return response.model_dump()
    
    # Get from cache or compute (10 min TTL)
    response_dict = await cache.get_or_compute(
        cache_key,
        compute,
        cache.config.image_list_ttl
    )
    
    return EvaluationResponse(**response_dict)


@router.get(
    "/projects/{project_id}/classes",
    response_model=List[PerClassMetrics],
    summary="Get per-class evaluation metrics"
)
async def get_class_metrics(
    project_id: int = Path(...),
    model_version: Optional[str] = Query(None),
    reviewed_only: bool = Query(True),
):
    """
    Get performance breakdown by class.
    Uses database aggregation for optimal performance.
    """
    cache = get_evaluation_cache()
    query_builder = EvaluationQueryBuilder()
    async def compute():
        return await query_builder.get_class_metrics(
            project_id=project_id,
            model_version=model_version,
            reviewed_only=reviewed_only
        )
        
    metrics = await cache.get_class_metrics(
        project_id=project_id,
        model_version=model_version,
        compute_fn=compute
    )
    
    return [PerClassMetrics(**m) for m in metrics]


@router.get(
    "/projects/{project_id}/summary",
    response_model=DatasetEvaluationSummary,
    summary="Quick dataset summary"
)
async def get_quick_summary(
    project_id: int = Path(...),
    model_version: Optional[str] = Query(None),
):
    """
    Lightweight endpoint for dashboard/overview.
    Returns only aggregated metrics using database aggregation.
    """
    
    cache = get_evaluation_cache()
    query_builder = EvaluationQueryBuilder()

    # Define compute function
    async def compute():
        summary = await query_builder.get_quick_summary(
            project_id=project_id,
            model_version=model_version
        )
        return summary.dict()
    
    # Get from cache or compute
    summary_dict = await cache.get_summary(
        project_id=project_id,
        model_version=model_version,
        compute_fn=compute
    )
    
    return DatasetEvaluationSummary(**summary_dict)


@router.get("/health", include_in_schema=False)
async def health_check():
    """Health check for monitoring"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ============================================================================
# EXPORT ENDPOINTS
# ============================================================================

@router.get(
    "/projects/{project_id}/export",
    summary="Export evaluation data"
)
async def export_evaluation(
    project_id: int = Path(...),
    format: str = Query("json", pattern="^(json|csv)$"),
    model_version: Optional[str] = Query(None),
):
    """
    Export complete evaluation data.
    Supports JSON and CSV formats with streaming for large datasets.
    """
    
    if format == "json":
        import orjson
        
        @sync_to_async
        def get_all_data():
            query_builder = EvaluationQueryBuilder()
            images, _ = query_builder.get_evaluated_images(
                project_id=project_id,
                model_version=model_version,
                page=1,
                page_size=10000  # Large batch for export
            )
            return [transform_image_to_response(img).dict() for img in images]
        
        async def generate():
            data = await get_all_data()
            yield orjson.dumps({"images": data, "count": len(data)})
        
        return StreamingResponse(
            generate(),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=evaluation_{project_id}.json"}
        )
    
    elif format == "csv":
        import csv
        import io
        
        @sync_to_async
        def generate_csv():
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow([
                'image_id', 'image_name', 'annotation_id', 'class_name',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'source', 'confidence', 'evaluation_status',
                'was_edited', 'edit_type', 'localization_iou'
            ])
            
            # Fetch data
            query_builder = EvaluationQueryBuilder()
            images, _ = query_builder.get_evaluated_images(
                project_id=project_id,
                model_version=model_version,
                page=1,
                page_size=10000
            )
            
            # Write rows
            for img in images:
                for ann in img.annotations.all():
                    try:
                        audit = ann.audit
                    except AnnotationAudit.DoesNotExist:
                        audit = None
                    
                    writer.writerow([
                        str(img.image_id),
                        img.name,
                        ann.annotation_uid,
                        ann.annotation_class.name,
                        ann.data[0], ann.data[1], ann.data[2], ann.data[3],
                        ann.annotation_source,
                        ann.confidence,
                        audit.evaluation_status if audit else None,
                        audit.was_edited if audit else False,
                        audit.edit_type if audit else None,
                        audit.localization_iou if audit else None,
                    ])
            
            return output.getvalue()
        
        csv_data = await generate_csv()
        
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=evaluation_{project_id}.csv"}
        )