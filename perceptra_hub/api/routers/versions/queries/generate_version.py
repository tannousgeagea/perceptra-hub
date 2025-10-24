import os
import uuid
import json
import zipfile
import shutil
import time
from fastapi import APIRouter, HTTPException, status, Header
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request
from fastapi import Response
from typing_extensions import Annotated
from fastapi.routing import APIRoute, APIRouter
from django.db import transaction
from django.core.files.base import ContentFile
from storages.backends.azure_storage import AzureStorage
from starlette.responses import FileResponse
from common_utils.azure_manager.core import AzureManager
from common_utils.data.annotation.core import save_annotations_into_txtfile
from projects.models import (
    Project, 
    ProjectImage, 
    Version, 
    VersionImage
)
from annotations.models import Annotation
from augmentations.models import VersionImageAugmentation
from common_utils.azure_manager.core import AzureManager
from common_utils.augmentation.core import AugmentationPipeline, PREDEFINED_AUGMENTATIONS
from common_utils.progress.core import track_progress

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
    "/versions/{project_id}/create", methods=["POST"], tags=["Versions"], status_code=status.HTTP_201_CREATED
    )
def create_version(
    response:Response,
    project_id: str,
    x_request_id: Annotated[Optional[str], Header()] = None,
    ):
    """
    Create a new version for a project by associating all reviewed images with the version.
    """
    try:
        task_id = x_request_id if x_request_id else str(uuid.uuid4())
        track_progress(task_id=task_id, percentage=0, status="Initializing ...")

        # Fetch the project
        project = Project.objects.filter(name=project_id)
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")

        # Determine the next version number
        project = project.first()
        last_version = Version.objects.filter(project=project).order_by('-version_number').first()
        next_version_number = last_version.version_number + 1 if last_version else 1
        images = ProjectImage.objects.filter(project=project, status="dataset", is_active=True)

        track_progress(task_id=task_id, percentage=0, status="Initiating Version")
        if not images.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="No reviewed images available to create a version"
            )

        # Start transaction
        with transaction.atomic():
            # Create a new version

            print(next_version_number)
            new_version = Version.objects.create(
                project=project,
                version_number=next_version_number,
                version_name=f"v{next_version_number}",
                created_at=datetime.now()
            )

            # Associate images with the new version
            version_images = [
                VersionImage(version=new_version, project_image=img)
                for img in images
            ]
            VersionImage.objects.bulk_create(version_images)

        track_progress(task_id=task_id, percentage=100, status="Initiating Done")
        track_progress(task_id=task_id, percentage=0, status="Initiating Images Generation")

        if not PREDEFINED_AUGMENTATIONS:
            return {"message": f"Version v{next_version_number} created successfully", "version_id": new_version.id, "version_number": new_version.version_number}
        
        output_dir = f"/tmp/augmented_dataset/{new_version.id}"
        aug_pipeline = AugmentationPipeline(output_dir=output_dir)
        for i, vi in enumerate(new_version.version_images.all()):
            proj_img = vi.project_image
            if proj_img.marked_as_null or proj_img.mode.mode.lower() != "train":
                track_progress(task_id=task_id, percentage=round((i / new_version.version_images.count()) * 100), status="Generating Images")
                continue

            image_field = proj_img.image.image_file.name
            cv_image = AzureManager.download_image_from_azure(image_field)

            print('annotation')
            ann_qs = Annotation.objects.filter(project_image=proj_img, is_active=True)
            ann_dict = {
                "bboxes": [ann.data for ann in ann_qs],
                "labels": [ann.annotation_class.class_id for ann in ann_qs]
            }
            augmented_files = aug_pipeline.apply_augmentations_policy(
                image=cv_image,
                annotations=ann_dict,
                annotation_type="detection",
                file_name_prefix=os.path.basename(proj_img.image.image_file.name),
                user_selected_augmentations=PREDEFINED_AUGMENTATIONS,
                multiplier=3,
            )

            for aug in augmented_files:
                aug_image_path = aug["image"]
                aug_annotation_path = aug["label"]

                aug_url = AzureManager.upload_image_to_azure(
                    local_file_path=aug_image_path, 
                    azure_path=f"versions/augmentations/{os.path.basename(aug_image_path)}")
                
                VersionImageAugmentation.objects.create(
                    version_image=vi,
                    augmentation_name="custom",
                    parameters=PREDEFINED_AUGMENTATIONS,
                    augmented_image_file=aug_url,
                    augmented_annotation=json.load(open(aug_annotation_path, "r")) if os.path.exists(aug_annotation_path) else ""
                )
            
            track_progress(task_id=task_id, percentage=round((i / new_version.version_images.count()) * 100), status="Generating Images")
        
        shutil.rmtree(output_dir, ignore_errors=True)
        track_progress(task_id=task_id, percentage=100, status="Completed")

        return {"message": f"Version v{next_version_number} created successfully", "version_id": new_version.id, "version_number": new_version.version_number}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )
