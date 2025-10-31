import time
import random
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from typing import Callable
from fastapi import Request
from fastapi import Response
from django.db.models import F
from fastapi.routing import APIRoute, APIRouter
from django.db import transaction
from projects.models import (
    Project, 
    ProjectImage, 
    Version, 
    VersionImage,
    ImageMode,
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
    responses={404: {"description": "Not found"}},
)

@router.api_route(
    "/projects/{project_id}/split", methods=["POST"], tags=["Projects"], status_code=status.HTTP_201_CREATED
    )
def create_version(
    response:Response,
    project_id: str,
    train_ratio:float,
    ):
    """
    Create a new version for a project by associating all reviewed images with the version.
    """
    try:
        # Fetch the project
        project = Project.objects.filter(name=project_id)
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")

        # Determine the next version number
        project = project.first()
        train_mode = ImageMode.objects.get(mode='train')
        valid_mode = ImageMode.objects.get(mode='valid')
        images = list(ProjectImage.objects.filter(project=project, status="dataset", mode=None))

        if not images:
            response.status_code = status.HTTP_200_OK
            return {
                "status_code": "no content",
                "detail": "All Images are splitted between Train and Valid Set"
            }
        
        # Start transaction
        with transaction.atomic():
            random.shuffle(images)

            # Split the images into train and validation sets
            split_index = int(len(images) * train_ratio)
            train_images = images[:split_index]
            val_images = images[split_index:]

            # Update the mode for train and validation images in bulk
            ProjectImage.objects.filter(pk__in=[image.pk for image in train_images]).update(mode=train_mode)
            ProjectImage.objects.filter(pk__in=[image.pk for image in val_images]).update(mode=valid_mode)
            
        return {"message": f"Train: {len(train_images)} | Valid: {len(val_images)}"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )
