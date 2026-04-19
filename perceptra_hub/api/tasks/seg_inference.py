# api/tasks/seg_inference.py

import json
import logging
import os
import uuid as uuid_lib

import numpy as np
from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    name="seg_inference:auto_segment",
    max_retries=3,
    queue="seg_inference",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
)
def auto_segment_task(
    self,
    session_id: str,
    image_id: int,
    model: str,
    device: str,
    precision: str,
    config: dict,
):
    """
    Celery task: auto-segment an image via the perceptra-seg inference service.
    Stores results in Redis and updates the SuggestionSession count.
    """
    from django.core.cache import cache
    from PIL import Image as PILImage
    from io import BytesIO
    import requests as req_lib

    from annotations.models import SuggestionSession
    from api.routers.suggestions.inference_client import SegInferenceClientSync
    from api.routers.suggestions.schemas import BoundingBox, Suggestion
    from projects.models import ProjectImage

    seg_url = os.environ.get("SEG_INFERENCE_URL", "").strip()
    if not seg_url:
        raise RuntimeError("SEG_INFERENCE_URL is not set — cannot run auto-segment")

    # 1. Load image from storage
    project_image = ProjectImage.objects.select_related(
        "image__storage_profile"
    ).get(id=image_id)

    storage_profile = project_image.image.storage_profile
    if storage_profile.backend == "local":
        base_path = storage_profile.config.get("base_path", "")
        pil_img = PILImage.open(f"{base_path}/{project_image.image.storage_key}")
    else:
        presigned_url = project_image.image.get_download_url(expiration=300)
        response = req_lib.get(presigned_url, timeout=30)
        response.raise_for_status()
        pil_img = PILImage.open(BytesIO(response.content)).convert("RGB")

    image_np = np.array(pil_img.convert("RGB"))

    # 2. Call inference service (synchronous)
    client = SegInferenceClientSync(
        base_url=seg_url,
        api_key=os.getenv("SEG_INFERENCE_API_KEY", ""),
        timeout_s=120.0,
    )
    outputs = client.segment_auto(image_np, config, model, device, precision)

    # 3. Filter by min_area and convert to Suggestion objects
    min_area = float(config.get("min_area", 0.001))
    suggestions = [
        Suggestion(
            suggestion_id=str(uuid_lib.uuid4()),
            bbox=BoundingBox(
                x=o.bbox[0],
                y=o.bbox[1],
                width=o.bbox[2],
                height=o.bbox[3],
            ),
            mask_rle=o.mask_rle,
            polygons=o.polygons,
            confidence=o.confidence,
            type="auto",
            status="pending",
        )
        for o in outputs
        if (o.bbox[2] * o.bbox[3]) >= min_area
    ]

    # 4. Store in Redis (matching SuggestionService cache key pattern)
    cache_key = f"suggestions:{session_id}"
    cache.set(cache_key, json.dumps([s.model_dump() for s in suggestions]), 3600)

    # 5. Update session count in DB
    SuggestionSession.objects.filter(suggestion_id=session_id).update(
        suggestions_generated=len(suggestions)
    )

    logger.info(
        "auto_segment_task completed: session=%s image=%s suggestions=%d",
        session_id,
        image_id,
        len(suggestions),
    )
    return len(suggestions)
