
import zipstream
from fastapi.responses import StreamingResponse
from io import BytesIO
import zipfile
import os
import os
import time
import django
django.setup()
from django.core.files.storage import default_storage
from datetime import datetime
from django.db.models import Q
from datetime import time as dtime
from typing import Callable, Optional, Dict, AnyStr, Any, List
from fastapi import Request
from fastapi import Response
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query
from fastapi.routing import APIRoute
from fastapi import status
from pathlib import Path
from pydantic import BaseModel

from images.models import Image
from .data import parse_query_list
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

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
@router.get("/images/download", tags=["Images"])
def stream_filtered_images_as_zip(
    limit: int = Query(100, ge=1, le=1000),
    name: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    query: Optional[List[str]] = Query(None),
):
    # Same queryset logic
    queryset = (
        Image.objects.prefetch_related("image_tags__tag", "projects")
        .select_related("sensorbox__edge_box")
        .order_by("-created_at")
    )

    if name:
        queryset = queryset.filter(image_name__icontains=name)

    if source:
        queryset = queryset.filter(
            sensorbox__edge_box__edge_box_location__icontains=source
        ) | queryset.filter(
            sensorbox__sensor_box_location__icontains=source
        ) | queryset.filter(
            source_of_origin__icontains=source
        )

    if tag:
        queryset = queryset.filter(image_tags__tag__name__icontains=tag)

    parsed_query = parse_query_list(query)
    if "tenant" in parsed_query:
        queryset = queryset.filter(
            sensorbox__edge_box__plant__tenant__name__icontains=parsed_query["tenant"]
        )
    if "location" in parsed_query:
        queryset = queryset.filter(
            sensorbox__sensor_box_location__icontains=parsed_query["location"]
        )
    if "tag" in parsed_query:
        queryset = queryset.filter(image_tags__tag__name__icontains=parsed_query["tag"])

    try:
        if "created_at" in parsed_query:
            queryset = queryset.filter(created_at=datetime.strptime(parsed_query["created_at"], "%Y-%m-%d"))
        if "created_at__gte" in parsed_query:
            queryset = queryset.filter(created_at__gte=datetime.strptime(parsed_query["created_at__gte"], "%Y-%m-%d"))
        if "created_at__lte" in parsed_query:
            queryset = queryset.filter(created_at__lte=datetime.strptime(parsed_query["created_at__lte"], "%Y-%m-%d"))
        if "created_at__range" in parsed_query:
            range_vals = parsed_query["created_at__range"].split(",")
            if len(range_vals) == 2:
                start = datetime.strptime(range_vals[0], "%Y-%m-%d")
                end = datetime.strptime(range_vals[1], "%Y-%m-%d")
                queryset = queryset.filter(created_at__range=(start, end))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format in query. Use YYYY-MM-DD or YYYY-MM-DD,YYYY-MM-DD")


    queryset = queryset[:limit]

    print(len(queryset))
    def zip_stream(queryset):
        """
        Given a queryset of Image objects, return a streaming zip generator.
        Uses zipstream for low-memory, efficient zip creation.
        """
        z = zipstream.ZipFile(mode='w', compression=zipstream.ZIP_DEFLATED, allowZip64=True)

        for image in queryset:
            image_path = image.image_file.name

            if not default_storage.exists(image_path):
                continue
            
            print(image_path)
            arcname = f"{image.image_name or image.image_id}{os.path.splitext(image_path)[-1]}"

            def file_iterator(path=image_path):
                with default_storage.open(path, 'rb') as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        yield chunk

            try:
                z.write_iter(arcname, file_iterator(image_path))
            except Exception as e:
                print(f"Error zipping image {image.image_id}: {e}")
            
        return z

    zip_filename = f"filtered_images_{queryset.count()}.zip"
    return StreamingResponse(
        zip_stream(queryset),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'}
    )
