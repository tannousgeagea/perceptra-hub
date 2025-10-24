import os
import time
import django
django.setup()
from fastapi import APIRouter
from fastapi import FastAPI, HTTPException, status
from fastapi import Request, Response
from pydantic import BaseModel
from fastapi.routing import APIRoute
from typing import Callable, Optional
from typing import List, Optional, Dict
from datetime import date
from datetime import datetime
from django.db.models.functions import TruncMonth
from django.db.models import Count
from fastapi import status as http_status

from projects.models import (
    Project,
    ProjectImage,
    ProjectMetadata,
    Version,
)

from annotations.models import (
    Annotation,
    AnnotationClass,
    AnnotationGroup,
)


from augmentations.models import (
    VersionImageAugmentation
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
    
class AugmentationTypeCount(BaseModel):
    name: str
    count: int


class AugmentationVersionCount(BaseModel):
    version_id: int
    version_number: str
    count: int


class AugmentationStats(BaseModel):
    total: int
    types: List[AugmentationTypeCount]
    version_distribution: List[AugmentationVersionCount]

router = APIRouter(
    route_class=TimedRoute,
)
@router.api_route(
    "/analytics/augmentationgroups", methods=["GET"], tags=["Analytics"])
def get_augmentation_groups(
    project_id:str
):
    try:
        try:
            project = Project.objects.get(name=project_id, is_active=True)
        except Project.DoesNotExist:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # All versions under this project
        versions = Version.objects.filter(project=project)
        version_ids = list(versions.values_list("id", flat=True))
        version_map = {v.id: (v.version_number or str(v.version_number)) for v in versions}

        # Get all VersionImageAugmentations for these versions
        augmentations = (
            VersionImageAugmentation.objects
            .filter(version_image__version__in=version_ids)
        )

        total = augmentations.count()

        # Type breakdown
        type_counts = (
            augmentations.values("augmentation_name")
            .annotate(count=Count("id"))
            .order_by("-count")
        )
        types: List[AugmentationTypeCount] = [
            AugmentationTypeCount(name=entry["augmentation_name"], count=entry["count"])
            for entry in type_counts
        ]

        # Version breakdown
        version_counts = (
            augmentations.values("version_image__version_id")
            .annotate(count=Count("id"))
        )
        version_distribution: List[AugmentationVersionCount] = [
            AugmentationVersionCount(
                version_id=entry["version_image__version_id"],
                version_number=f"v{version_map.get(entry['version_image__version_id'], 'unknown')}",
                count=entry["count"]
            )
            for entry in version_counts
        ]

        return AugmentationStats(
            total=total,
            types=types,
            version_distribution=version_distribution
        )
    
    except HTTPException as e:
        raise  e
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= str(e),
        )