
import cv2
import time
import random
import numpy as np
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from typing import Callable
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


router = APIRouter(
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)

REMOTE_INFERENCE_URL = "http://10.7.0.6:8081/infer"

@router.api_route(
    "/analyse", methods=["POST"], tags=["Analysis"]
    )
async def analyse_image(file: UploadFile = File(...)):
    try:
        # contents = await file.read()
        # nparr = np.fromstring(contents, np.uint8)
        # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # if img is None:
        #     raise HTTPException(status_code=400, detail="Invalid image file")
        
        files = {"image": (file.filename, await file.read(), file.content_type)}
        response = requests.post(REMOTE_INFERENCE_URL, files=files)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Error in inference service")

        predictions = response.json()["predictions"]
        return {"predictions": predictions}
        
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )