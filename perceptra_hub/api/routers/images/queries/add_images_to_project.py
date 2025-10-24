
import time
import uuid
from fastapi import Request, Response
from fastapi.routing import APIRoute
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from django.db import transaction
from typing import Callable
from django.core.exceptions import ObjectDoesNotExist
from images.models import Image
from projects.models import Project, ProjectImage
from common_utils.jobs.utils import assign_uploaded_image_to_batch


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

# Pydantic schema for request payload
class AddImagesRequest(BaseModel):
    project_id: int
    image_ids: List[str]

@router.post("/projects/{project_id}/add-images")
def add_images_to_project(project_id: int, request: AddImagesRequest):
    """
    Adds selected images to a specified project.
    """
    try:
        project = Project.objects.get(id=project_id)
    except ObjectDoesNotExist:
        raise HTTPException(status_code=404, detail="Project not found.")

    images = Image.objects.filter(image_id__in=request.image_ids)

    if not images.exists():
        raise HTTPException(status_code=404, detail="No valid images found.")

    # Transactional bulk creation
    with transaction.atomic():
        created_links = []
        for image in images:
            # Check if already linked
            if ProjectImage.objects.filter(project=project, image=image).exists():
                continue
            project_image = ProjectImage(
                project=project,
                image=image,
                status='unannotated'
            )
            created_links.append(project_image)
        
        ProjectImage.objects.bulk_create(created_links)

        batch_id = str(uuid.uuid4())
        for project_image in created_links:
            assign_uploaded_image_to_batch(
                project_image=project_image,
                batch_id=batch_id
            )

    return {"message": f"{len(created_links)} image(s) added to project {project.name}."}
