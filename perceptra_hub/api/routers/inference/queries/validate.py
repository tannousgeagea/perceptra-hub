# api/validation.py

import time
import uuid
from fastapi import Header
from fastapi import APIRouter, HTTPException
from ml_models.models import ModelVersion
from projects.models import VersionImage, ImageMode
from inferences.models import PredictionImageResult, PredictionOverlay
from common_utils.inference.utils import run_inference
from django.db import transaction
from typing_extensions import Annotated
from typing import Optional
from common_utils.progress.core import track_progress

router = APIRouter()

@router.post("/validate-model/{model_version_id}")
def validate_model(
    model_version_id: int,
    x_request_id: Annotated[Optional[str], Header()] = None,
    ):
    task_id = x_request_id if x_request_id else str(uuid.uuid4())
    track_progress(task_id=task_id, percentage=0, message="Initiating Validation ...", status="pending")
    try:
        try:
            model_version = ModelVersion.objects.get(id=model_version_id)
        except ModelVersion.DoesNotExist:
            track_progress(task_id=task_id, percentage=0, message=f"Failed: Model Version not Found", status="failed")
            raise HTTPException(status_code=404, detail="Model version not found")

        version = model_version.dataset_version
        mode = ImageMode.objects.get(mode="valid")
        version_images = VersionImage.objects.filter(version=version, project_image__mode=mode)
        created_results = 0

        track_progress(task_id=task_id, percentage=0, message="Started Validation", status="pending")
        for i, vi in enumerate(version_images):
            image = vi.project_image.image
            dataset_version = vi.version 

            # Skip if already validated
            if PredictionImageResult.objects.filter(
                model_version=model_version,
                dataset_version=dataset_version,
                image=image
            ).exists():
                continue

            start_time  = time.time()
            predictions = run_inference(image=image.image_file.name, model_version_id=model_version_id, confidence_threshold=0.01)
            inference_time = time.time() - start_time

            with transaction.atomic():
                result = PredictionImageResult.objects.create(
                    model_version=model_version,
                    dataset_version=dataset_version,
                    image=image,
                    inference_time=round(inference_time, 4),
                )

                overlay_objs = [
                    PredictionOverlay(
                        prediction_result=result,
                        class_id=pred["class_id"],
                        class_label=pred["class_label"],
                        confidence=pred["confidence"],
                        bbox=pred.get("xyxyn"),
                        mask=pred.get("mask"),
                        overlay_type=pred.get("overlay_type", "bbox"),
                    )
                    for pred in predictions
                ]

                PredictionOverlay.objects.bulk_create(overlay_objs)
                created_results += 1
            
            track_progress(task_id=task_id, percentage=round((i / version_images.count()) * 100), message="Running Validation ...", status="running")

        track_progress(task_id=task_id, percentage=100, message="✅ Validation completed successfully", status="completed")
        return {
            "message": f"Validation completed for model version {model_version_id}",
            "total_images_processed": version_images.count(),
            "new_results_created": created_results
        }

    except Exception as e:
        track_progress(task_id=task_id, message=f"Internal Server Error: {e}", status="failed")
        raise HTTPException(
            status_code=500,
            detail=f"{e}"
        )