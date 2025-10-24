
import os
import math
import uuid
import time
import django
import shutil
django.setup()
from datetime import datetime, timedelta
from datetime import date, timezone
from typing import Callable, Optional, Dict, AnyStr, Any, List
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import Depends, Form, Body
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi import status
from pathlib import Path
from django.db import transaction
from pydantic import BaseModel

from projects.models import (
    Project,
    ProjectImage,
)

from annotations.models import (
    Annotation,
    AnnotationClass,
    AnnotationGroup,
    AnnotationType
)

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
)

class AnnotationData(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float
    label: str
    color: str

@router.api_route(
    "/annotations/{id}", methods=["DELETE"], tags=["Annotations"])
def delete_annotation(id:str):
    try:
        if not Annotation.objects.filter(annotation_uid=id).exists():
            raise HTTPException(
                status_code=404, detail=f"Annotation id {id} not Found")
        
        annotation = Annotation.objects.filter(annotation_uid=id).first()
        annotation.is_active = False
        annotation.save()
        
        return {"message": "Annotations deleted successfully."}
        
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))