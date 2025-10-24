# api/validation_result.py

from fastapi import APIRouter, HTTPException, Query
from ml_models.models import ModelVersion
from inferences.models import PredictionImageResult, PredictionOverlay
from annotations.models import Annotation
from images.models import Image
from projects.models import ProjectImage

router = APIRouter()

def convert_bbox(bbox, width:int, height:int):
    """
    Convert normalized [x1, y1, x2, y2] to x, y, width, height in absolute pixels (assumes 1.0 if not scaled)
    """
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox

    if width is None:
        width = 400
    
    if height is None:
        height = 300

    return {
        "x": int(x1 * width),
        "y": int(y1 * height),
        "width": int((x2 - x1) * width),
        "height": int((y2 - y1) * height)
    }

@router.get("/model-version/{model_version_id}/validation-images")
def get_validation_images(
    model_version_id: int,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
    ):
    try:
        model_version = ModelVersion.objects.get(id=model_version_id)
    except ModelVersion.DoesNotExist:
        raise HTTPException(status_code=404, detail="ModelVersion not found")

    total = PredictionImageResult.objects.filter(model_version=model_version).count()
    prediction_results = (
        PredictionImageResult.objects
        .filter(model_version=model_version)
        .select_related('image')
        .prefetch_related('overlays')
        .order_by('id')[offset:offset + limit]
    )
    
    results = []

    for pr in prediction_results:
        image = pr.image
        pi = ProjectImage.objects.filter(
            project=model_version.model.project,
            image=image
        ).first()

        # Collect predictions
        prediction_boxes = [
            {
                **convert_bbox(p.bbox, width=image.width, height=image.height),
                "label": p.class_label,
                "confidence": p.confidence,
                "type": "prediction"
            }
            for p in pr.overlays.all()
            if p.bbox
        ]

        # Collect ground-truths
        gt_annotations = Annotation.objects.filter(
            project_image=pi,
            is_active=True
        )

        ground_truth_boxes = [
            {
                **convert_bbox(gt.data, width=image.width, height=image.height),
                "label": gt.annotation_class.name,
                "confidence": 1.0,
                "type": "groundTruth"
            }
            for gt in gt_annotations
            if isinstance(gt.data, list) and len(gt.data) == 4
        ]

        all_boxes = prediction_boxes + ground_truth_boxes
        avg_conf = round(
            sum(p["confidence"] for p in prediction_boxes) / len(prediction_boxes), 2
        ) if prediction_boxes else 0.0

        results.append({
            "id": image.id,
            "width": image.width,
            "height": image.height,
            "original": image.image_file.url,  # or image.image_url if custom field
            "confidence": avg_conf,
            "boundingBoxes": all_boxes
        })

    return {
        "count": total,
        "limit": limit,
        "offset": offset,
        "results": results
    }
