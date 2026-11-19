
import os
import cv2
import random
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from asgiref.sync import sync_to_async
from ml_models.models import ModelVersion
from api.dependencies import RequestContext, require_permission
from datetime import datetime
from typing import Optional
from django.db.models import F
from fastapi.routing import APIRouter
from django.db import transaction
from fastapi import UploadFile, File
import requests


INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", os.getenv("REMOTE_INFERENCE_URL", ""))

router = APIRouter(
    responses={404: {"description": "Not found"}},
)

@router.api_route(
    "/infer/{model_version_id}", methods=["POST"],
    )
async def infer(
    model_version_id:int,
    file: UploadFile = File(...),
    confidence_threshold:Optional[float] = 0.25,
    max_detections:Optional[int] = 100,
    # Running models is the org "operate" permission (operator and up).
    ctx: RequestContext = Depends(require_permission("operate")),
):
    owned = await sync_to_async(
        ModelVersion.objects.filter(id=model_version_id, model__organization=ctx.organization).exists
    )()
    if not owned:
        raise HTTPException(status_code=404, detail="Model version not found")
    files = {"file": (file.filename, await file.read(), file.content_type)}
    params = {
            "confidence_threshold": confidence_threshold,
            "max_detections": max_detections,
    }
    response = requests.post(f"{INFERENCE_SERVICE_URL}/v1/infer/{model_version_id}", files=files, params=params)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Error in inference service: {response.text}")

    return response.json()