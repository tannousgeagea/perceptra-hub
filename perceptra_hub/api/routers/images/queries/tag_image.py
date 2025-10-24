import os
import math
import uuid
import time
import django
import shutil
django.setup()
from django.db.models import Q
from datetime import datetime, timedelta
from datetime import time as dtime
from datetime import date, timezone
from typing import Callable, Optional, Dict, AnyStr, Any
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi import Body, Depends
from pathlib import Path
from pydantic import BaseModel
from django.db import transaction
from images.models import Image, Tag, ImageTag
from django.shortcuts import get_object_or_404
from api.routers.auth.queries.dependencies import get_current_user

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

class TagCreateRequest(BaseModel):
    tag_name: str

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

@router.api_route(
    "/images/{image_id}/tags", methods=["POST"], tags=["Images"]
)
@transaction.atomic
def tag_image(
    image_id: str,
    data: TagCreateRequest = Body(...),
    user = Depends(get_current_user),
):
    image = get_object_or_404(Image, image_id=image_id)
    tag, _ = Tag.objects.get_or_create(name=data.tag_name)

    image_tag, created = ImageTag.objects.get_or_create(
        image=image, tag=tag,
        defaults={"tagged_by": user}
    )

    if not created:
        raise HTTPException(status_code=400, detail="Tag already exists on image")

    return {"detail": f"Tag '{tag.name}' added to image '{image.image_name}'"}