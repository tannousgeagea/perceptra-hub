
import os
import io
import uuid
import time
import yaml
from io import StringIO
import zipstream
from PIL import Image
import threading
from queue import Queue
from typing_extensions import Annotated
from fastapi import APIRouter, HTTPException, Query, Header
from fastapi.routing import APIRoute
from fastapi import Request, Response
from typing import Callable, Optional, Literal
from django.core.files.base import ContentFile
from starlette.responses import StreamingResponse, FileResponse, RedirectResponse
from projects.models import Version, VersionImage
from annotations.models import Annotation, AnnotationGroup
from django.core.files.storage import default_storage
from common_utils.data.annotation.core import format_annotation, read_annotation
from common_utils.progress.core import track_progress

def compress_image(path, quality=75):
    """Compress image and return bytes."""
    with default_storage.open(path, 'rb') as f:
        img = Image.open(f)
        img_io = io.BytesIO()
        img.convert('RGB').save(img_io, format='JPEG', optimize=True, quality=quality)
        img_io.seek(0)
        return img_io.read()

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


def get_cached_zip_path(cached_zip_path) -> str:
    """Return the cached zip file path if it exists, else an empty string."""
    if os.getenv("CACHE_ZIP_PATH") == "azure":
        if default_storage.exists(cached_zip_path):
            return cached_zip_path
    else:
        if os.path.exists(cached_zip_path):
            return cached_zip_path
    return ""


def generate_data_yaml_content(class_names: list) -> bytes:
    data = {
        "train": "train/images",
        "val": "valid/images",
        "nc": len(class_names),
        "names": class_names
    }
    buffer = StringIO()
    yaml.dump(data, buffer)
    return buffer.getvalue().encode("utf-8")

def generate_zip_stream(version: Version, annotation_format: str, task_id:str):
    """
    Generate a streaming zip file for a given version with annotations converted to the desired format.
    This function uses zipstream to build the zip file on the fly.

    Args:
        version (Version): The version instance.
        annotation_format (str): Desired annotation format (e.g., "yolo", "custom", etc.).

    Returns:
        zipstream.ZipFile: An iterable zip file stream.
    """
    z = zipstream.ZipFile(mode='w', compression=zipstream.ZIP_DEFLATED, allowZip64=True)
    # z.write("start.txt", "Processing started...")
    version_images = VersionImage.objects.filter(version=version)

    for i, vi in enumerate(version_images):
        proj_img = vi.project_image
        prefix = proj_img.mode.mode if proj_img.mode else "default"
        image_path = proj_img.image.image_file.name
        image_filename = f"{os.path.basename(proj_img.image.image_file.name)}.jpg"
        def file_iterator(path):
            with default_storage.open(path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk
        
        if not default_storage.exists(image_path):
            continue

        try:
            # compressed_image = compress_image(image_path, quality=65)
            print(image_path)
            file_iter = file_iterator(image_path)
            z.write_iter(f"{prefix}/images/{image_filename}", file_iter)
        except Exception as e:
            print(f"Error compressing or adding original image {image_filename}: {e}")

        annotations = Annotation.objects.filter(project_image=proj_img, is_active=True)
        annotation_str = "".join([format_annotation(ann, format=annotation_format) for ann in annotations])
        annotation_bytes = annotation_str.encode('utf-8')
        z.writestr(f"{prefix}/labels/{os.path.basename(proj_img.image.image_file.name)}.txt", annotation_bytes)

        augmentations = vi.augmentations.all()
        for aug in augmentations:
            # Create filenames that include the augmentation name.
            aug_image_filename = f"{os.path.basename(aug.augmented_image_file.name)}.jpg"
            aug_annotation_filename = f"{os.path.basename(aug.augmented_image_file.name)}.txt"
            
            if aug.augmented_image_file:
                try:
                    z.write_iter(f"{prefix}/images/{aug_image_filename}", file_iterator(aug.augmented_image_file.name))
                except Exception as e:
                    print(f"Error adding augmented image for {os.path.basename(proj_img.image.image_file.name)}: {e}")
            
            if aug.augmented_annotation:
                # Convert annotation dictionary to string using read_annotation helper.
                annotation_str = "".join([
                    read_annotation(bbox=bbox, label=label, format=annotation_format)
                    for bbox, label in zip(aug.augmented_annotation['bboxes'], aug.augmented_annotation['labels'])
                ])
                z.writestr(f"{prefix}/labels/{aug_annotation_filename}", annotation_str.encode("utf-8"))
        track_progress(task_id=task_id, percentage=round((i / version_images.count()) * 100), status="Zipping Files ...")

    if annotation_format == "yolo":
        annotation_group = AnnotationGroup.objects.filter(project=version.project).first()
        classes = annotation_group.classes.all().order_by('class_id')
        class_names = [cls.name for cls in classes]
        yaml_content = generate_data_yaml_content(class_names)
        z.writestr(f"data.yaml", yaml_content)

    return z

def generate_and_upload_streaming(version: Version, format: str, blob_client):
    """Generate zip and upload simultaneously without storing locally"""
    
    class StreamingUploader:
        def __init__(self, blob_client, chunk_size=8*1024*1024):  # 8MB chunks
            self.blob_client = blob_client
            self.chunk_size = chunk_size
            self.buffer = io.BytesIO()
            self.chunk_id = 0
            self.block_ids = []
            
        def write(self, data):
            self.buffer.write(data)
            if self.buffer.tell() >= self.chunk_size:
                self.flush_chunk()
                
        def flush_chunk(self):
            if self.buffer.tell() == 0:
                return
                
            self.buffer.seek(0)
            block_id = f"{self.chunk_id:06d}"
            
            # Upload chunk with retry
            for attempt in range(3):
                try:
                    self.blob_client.stage_block(
                        block_id=block_id,
                        data=self.buffer.read()
                    )
                    self.block_ids.append(block_id)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    time.sleep(2 ** attempt)
            
            self.buffer = io.BytesIO()
            self.chunk_id += 1
            print(f"Uploaded chunk {self.chunk_id}")
            
        def finalize(self):
            self.flush_chunk()  # Upload any remaining data
            if self.block_ids:
                self.blob_client.commit_block_list(self.block_ids)
    
    uploader = StreamingUploader(blob_client)
    zip_stream = generate_zip_stream(version, format, task_id="streaming")
    
    try:
        for chunk in zip_stream:
            uploader.write(chunk)
        uploader.finalize()
    except Exception as e:
        print(f"Upload failed: {e}")
        raise


@router.get("/versions/{version_id}/download", tags=["Versions"])
def download_version(
    version_id: int, 
    format:Literal["yolo", "custom", "coco"] = Query("yolo", description="Desired annotation format"),
    x_request_id: Annotated[Optional[str], Header()] = None,
    ):
    """
    On-demand endpoint to download a version's images with annotations in a preferred format.
    
    - The version images remain stored in Azure.
    - Annotations are converted to the requested format on the fly.
    - The zip archive is built dynamically and streamed back to the user.
    
    **Query Parameters:**
      - **format**: Desired annotation format (default is "yolo").
    
    **Response:**
      - A streaming zip file containing:
          - For each image: the image file (under `{mode}/images/`) and its corresponding annotation file (under `{mode}/labels/`).
    """
    try:
        version = Version.objects.get(id=version_id)
    except Version.DoesNotExist:
        raise HTTPException(status_code=404, detail="Version not found")
    
    task_id = x_request_id if x_request_id else str(uuid.uuid4())
    filename = f"{version.project.name}.v{version.version_number}.{format}.zip"

    # Check if already exists
    version_zip_rel_path = f"versions/{filename}"
    if default_storage.exists(version_zip_rel_path):
        return {"url": default_storage.url(version_zip_rel_path)}

    track_progress(task_id=task_id, percentage=0, status="Starting upload ... might take a while")
    blob_client = default_storage.client.get_blob_client(
        blob=version_zip_rel_path
    )
    
    # Stream directly to Azure without local storage
    generate_and_upload_streaming(version, format, blob_client)
    
    # Update version record
    version.version_file = version_zip_rel_path
    version.save(update_fields=["version_file"])
    track_progress(task_id=task_id, percentage=100, status="Completed")
    return {"url": version.version_file.url}

