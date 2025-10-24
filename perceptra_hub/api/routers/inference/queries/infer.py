
import os
import cv2
import time
import random
import numpy as np
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request
from fastapi import Response
from django.db.models import F
from fastapi.routing import APIRoute, APIRouter
from django.db import transaction
from fastapi import UploadFile, File
import requests

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


REMOTE_INFERENCE_URL = os.getenv("REMOTE_INFERENCE_URL")

router = APIRouter(
    route_class=TimedRoute,
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
):
    files = {"file": (file.filename, await file.read(), file.content_type)}
    params = {
            "confidence_threshold": confidence_threshold,
            "max_detections": max_detections,
    }
    response = requests.post(f"{REMOTE_INFERENCE_URL}/api/v1/infer/{model_version_id}", files=files, params=params)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Error in inference service: {response.text}")

    return response.json()