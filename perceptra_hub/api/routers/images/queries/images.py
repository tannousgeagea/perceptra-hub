"""
FastAPI routes for image management and upload.
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List
from uuid import UUID
from datetime import datetime
import hashlib
import logging
import uuid
from io import BytesIO
from PIL import Image as PILImage
from asgiref.sync import sync_to_async

import django
django.setup()

from django.db import models

from api.dependencies import get_current_user, get_current_organization, RequestContext, get_request_context, bypass_auth_dev
from storage.models import StorageProfile
from storage.services import get_storage_adapter_for_profile, get_default_storage_adapter
from images.models import Image, Tag, ImageTag
from organizations.models import Organization
from django.contrib.auth import get_user_model
from common_utils.image.utils import parse_search_query, apply_image_filters
from common_utils.jobs.utils import assign_uploaded_image_to_batch

User = get_user_model()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images")


# ============= Helper Functions =============

def calculate_checksum(file_data: bytes) -> str:
    """Calculate SHA-256 checksum of file."""
    return hashlib.sha256(file_data).hexdigest()


def get_image_dimensions(file_data: bytes) -> tuple[int, int]:
    """Extract image dimensions using PIL."""
    try:
        img = PILImage.open(BytesIO(file_data))
        return img.size  # Returns (width, height)
    except Exception as e:
        logger.error(f"Failed to get image dimensions: {e}")
        return (0, 0)


def get_image_format(filename: str) -> str:
    """Extract file format from filename."""
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'unknown'


def generate_storage_key(organization: Organization, filename: str, project_id: Optional[UUID] = None) -> str:
    """
    Generate storage key for image.
    
    Format: org-{slug}/images/{year}/{month}/{uuid}_{filename}
    Or with project: org-{slug}/projects/{project-id}/images/{year}/{month}/{uuid}_{filename}
    """
    import uuid
    from datetime import datetime
    
    now = datetime.now()
    year = now.strftime('%Y')
    month = now.strftime('%m')
    unique_id = str(uuid.uuid4())[:8]
    
    # Sanitize filename
    safe_filename = filename.replace(' ', '_').replace('/', '_')
    
    if project_id:
        return f"org-{organization.slug}/projects/{project_id}/images/{year}/{month}/{unique_id}_{safe_filename}"
    else:
        return f"org-{organization.slug}/images/{year}/{month}/{unique_id}_{safe_filename}"


# ============= Image Upload Endpoints =============
@sync_to_async
def process_image_upload(
    ctx: RequestContext,
    file_data: bytes,
    filename: str,
    content_type: Optional[str],
    file_format: str,
    name: Optional[str],
    image_id: Optional[str],
    project_id: Optional[UUID],
    tag_names: List[str],
    source_of_origin: Optional[str],
    storage_profile_id: Optional[UUID],
    batch_id: Optional[str] = Query(None),
):
    """
    Synchronous function to process image upload.
    """
    file_size = len(file_data)
    
    # Calculate checksum
    checksum = calculate_checksum(file_data)
    
    # Check for duplicate (same checksum in same organization)
    existing = Image.objects.filter(
        organization=ctx.organization,
        checksum=checksum
    ).first()
    
    if existing:
        logger.warning(f"Duplicate image detected: {checksum}")
        return {
            "duplicate": True,
            "image": existing
        }
    
    # Get image dimensions
    width, height = get_image_dimensions(file_data)
    
    if width == 0 or height == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read image dimensions. File may be corrupted."
        )
    
    # Get storage profile
    if storage_profile_id:
        try:
            storage_profile = StorageProfile.objects.get(
                id=storage_profile_id,
                organization=ctx.organization,
                is_active=True
            )
        except StorageProfile.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Storage profile {storage_profile_id} not found"
            )
    else:
        # Use default storage profile
        storage_profile = StorageProfile.objects.filter(
            organization=ctx.organization,
            is_default=True,
            is_active=True
        ).first()
        
        if not storage_profile:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No default storage profile configured for organization"
            )
    
    if not image_id:
        import uuid
        image_id = str(uuid.uuid4())
    
    # Generate storage key
    storage_key = generate_storage_key(
        ctx.organization,
        filename,
        project_id
    )
    
    # Upload to storage
    adapter = get_storage_adapter_for_profile(storage_profile)
    
    adapter.upload_file(
        BytesIO(file_data),
        storage_key,
        content_type=content_type or 'image/jpeg',
        metadata={
            'organization': ctx.organization.slug,
            'uploaded_by': ctx.user.username,
            'original_filename': filename
        }
    )
    
    logger.info(f"Uploaded file to storage: {storage_key}")
    
    # Create Image record
    image = Image.objects.create(
        organization=ctx.organization,
        storage_profile=storage_profile,
        storage_key=storage_key,
        name=name or filename,
        image_id=image_id,
        original_filename=filename,
        file_format=file_format,
        file_size=file_size,
        width=width,
        height=height,
        checksum=checksum,
        source_of_origin=source_of_origin or 'upload',
        uploaded_by=ctx.user
    )
    
    logger.info(f"Created image record: {image.id}")
    # Associate with project if provided
    if project_id:
        from projects.models import Project, ProjectImage
        try:
            project = Project.objects.get(
                project_id=project_id,
                organization=ctx.organization
            )
            
            project_image, pi_created = ProjectImage.objects.get_or_create(
                image=image,
                project=project,
                defaults={
                    'added_by': ctx.user,
                    'job_assignment_status': 'waiting'
                }
            )
            
            if pi_created:
                logger.info(f"Associated image {image.id} with project {project_id}")
                
                logger.warning(f"Batch ID: {batch_id}")
                assigned_job = assign_uploaded_image_to_batch(
                    project_image=project_image,
                    batch_id=batch_id,
                )
                
                if assigned_job:
                    logger.info(f"Auto-assigned image to job {assigned_job.id}")
        except Project.DoesNotExist:
            logger.warning(f"Project {project_id} not found")
    
    # Add tags if provided
    if tag_names:
        for tag_name in tag_names:
            tag, created = Tag.objects.get_or_create(
                organization=ctx.organization,
                name=tag_name
            )
            
            ImageTag.objects.create(
                image=image,
                tag=tag,
                tagged_by=ctx.user
            )
        
        logger.info(f"Added {len(tag_names)} tags to image {image.id}")
    
    return {
        "duplicate": False,
        "image": image
    }
    
@sync_to_async
def get_dowload_url(image:Image, expiration:int=3600):
    return image.get_download_url(expiration=expiration)

@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload Image",
    description="Upload an image to organization's default storage"
)
async def upload_image(
    file: UploadFile = File(..., description="Image file to upload"),
    name: Optional[str] = Form(None, description="Human-readable name"),
    image_id: Optional[str] = Form(None, description="Optional custom image ID (UUID format)"),  # ← ADD THIS
    project_id: Optional[UUID] = Form(None, description="Optional project ID"),
    tags: Optional[str] = Form(None, description="Comma-separated tag names"),
    source_of_origin: Optional[str] = Form(None, description="Source of image"),
    storage_profile_id: Optional[UUID] = Form(None, description="Specific storage profile to use"),
    batch_id:Optional[str] = Form(None, description="batch Id where image be assigned to one Job"),
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Upload an image to organization's storage.
    
    The image will be stored in the default storage profile unless specified.
    Image metadata will be saved to database for tracking.
    """
    try:
        # Validate file type
        allowed_formats = ['jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp', 'webp']
        file_format = get_image_format(file.filename)
        
        if file_format not in allowed_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {file_format}. Allowed: {', '.join(allowed_formats)}"
            )
        
        if image_id:
            # Validate UUID format
            try:
                uuid.UUID(image_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid image_id format. Must be a valid UUID."
                )
            
            # Check if image_id already exists
            if await sync_to_async(Image.objects.filter(
                organization=ctx.organization,
                image_id=image_id
            ).exists)():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Image with image_id '{image_id}' already exists"
                )
        
        # Read file data
        file_data = await file.read()
        file_size = len(file_data)
        
        # Validate file size (max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {max_size / (1024*1024)}MB"
            )
        
        # Parse tags
        tag_names = [t.strip() for t in tags.split(',') if t.strip()] if tags else []
        
        # Process upload in sync context
        result = await process_image_upload(
            ctx=ctx,
            file_data=file_data,
            filename=file.filename,
            content_type=file.content_type,
            file_format=file_format,
            name=name,
            image_id=image_id,
            project_id=project_id,
            tag_names=tag_names,
            source_of_origin=source_of_origin,
            storage_profile_id=storage_profile_id,
            batch_id=batch_id,
        )
        
        image = result["image"]
        
        if result["duplicate"]:
            return {
                "message": "Image already exists",
                "image_id": str(image.id),
                "duplicate": True
            }
        
        # Generate download URL
        # download_url = image.get_download_url(expiration=3600)
        download_url = await get_dowload_url(image, expiration=3600)
        
        return {
            "message": "Image uploaded successfully",
            "image_id": str(image.image_id),
            "name": image.name,
            "storage_key": image.storage_key,
            "file_size": image.file_size,
            "file_size_mb": round(image.file_size_mb, 2),
            "dimensions": {
                "width": image.width,
                "height": image.height
            },
            "aspect_ratio": round(image.aspect_ratio, 2),
            "megapixels": round(image.megapixels, 2),
            "format": image.file_format,
            "checksum": image.checksum,
            "download_url": download_url,
            "created_at": image.created_at.isoformat(),
            "duplicate": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Image upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image upload failed: {str(e)}"
        )


@router.post(
    "/upload/batch",
    status_code=status.HTTP_201_CREATED,
    summary="Batch Upload Images",
    description="Upload multiple images at once"
)
async def batch_upload_images(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    project_id: Optional[UUID] = Form(None),
    source_of_origin: Optional[str] = Form(None),
    ctx: RequestContext = Depends(get_request_context)
):
    """Upload multiple images in a batch."""
    
    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 files per batch upload"
        )
    
    results = {
        "total": len(files),
        "successful": 0,
        "failed": 0,
        "duplicates": 0,
        "images": []
    }
    
    
    batch_id = uuid.uuid4()
    for file in files:
        try:
            # Reset file pointer
            await file.seek(0)
            
            # Upload single image (reuse logic)
            result = await upload_image(
                file=file,
                name=None,
                project_id=project_id,
                tags=None,
                source_of_origin=source_of_origin,
                storage_profile_id=None,
                ctx=ctx,
                batch_id=batch_id,
            )
            
            if result.get('duplicate'):
                results['duplicates'] += 1
            else:
                results['successful'] += 1
            
            results['images'].append({
                "filename": file.filename,
                "status": "success",
                "image_id": result['image_id'],
                "duplicate": result.get('duplicate', False)
            })
            
        except Exception as e:
            results['failed'] += 1
            results['images'].append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"Failed to upload {file.filename}: {e}")
    
    return results


# ============= Image Retrieval Endpoints =============

@router.get(
    "",
    summary="List Images",
    description="List all images for the organization"
)
async def list_images(
    ctx: RequestContext = Depends(get_request_context),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    project_id: Optional[UUID] = Query(None),
    q: Optional[str] = Query(
        None,
        description="Search query (e.g., 'tag:car min-width:1920 class:vehicle')",
        alias="q"
    ),
    tag: Optional[str] = Query(None),
    search: Optional[str] = Query(None)
):
    """
    List images with advanced filtering.
    
    **Query Syntax:**
    - `tag:name` - Filter by tag
    - `class:name` - Filter by annotation class
    - `split:train|val|test` - Filter by dataset split
    - `filename:text` - Filter by filename
    - `min-width:1920` - Minimum width
    - `max-width:1920` - Maximum width
    - `min-height:1080` - Minimum height
    - `max-height:1080` - Maximum height
    - `min-annotations:5` - Minimum annotation count
    - `max-annotations:10` - Maximum annotation count
    - `job:job-name` - Filter by annotation job
    - `sort:size|name|date|width|height|annotations` - Sort results
    
    **Examples:**
    - `tag:car tag:red min-width:1920`
    - `class:vehicle split:train filename:IMG`
    - `min-annotations:5 sort:size`
    """
    @sync_to_async
    def get_images_list(
        organization: Organization,
        skip: int,
        limit: int,
        project_id: Optional[UUID],
        query: Optional[str],
        legacy_tag: Optional[str],
        legacy_search: Optional[str]
    ):
        """
        Synchronous function to get images list.
        """
        queryset = Image.objects.filter(organization=organization)
        
        # Filter by project (through ProjectImage relationship)
        if project_id:
            queryset = queryset.filter(project_images__project_id=project_id)
        
        # Parse and apply query filters
        if query:
            filters = parse_search_query(query)
            queryset = apply_image_filters(queryset, filters)
        else:
            # Legacy filters for backwards compatibility
            if legacy_tag:
                queryset = queryset.filter(tags__name=legacy_tag)
            
            if legacy_search:
                queryset = queryset.filter(
                    models.Q(name__icontains=legacy_search) |
                    models.Q(original_filename__icontains=legacy_search)
                )
            
            # Default sort
            queryset = queryset.order_by('-created_at')
        
        # Get total count
        total = queryset.distinct().count()
        
        # Get paginated results
        images = list(
            queryset.select_related(
                'storage_profile',
                'uploaded_by'
            ).prefetch_related('tags').distinct()[skip:skip + limit]
        )
        
        return {
            "total": total,
            "images": images,
            "skip": skip,
            "limit": limit
        }
    
    result = await get_images_list(
        ctx.organization,
        skip,
        limit,
        project_id,
        q,
        tag,
        search
    )
    
    return {
        "total": result["total"],
        "page": (result["skip"] // result["limit"]) + 1,
        "page_size": result["limit"],
        "images": [
            {
                "id": str(img.id),
                "image_id": img.image_id,
                "name": img.name,
                "original_filename": img.original_filename,
                "file_format": img.file_format,
                "file_size": img.file_size,
                "file_size_mb": round(img.file_size_mb, 2),
                "width": img.width,
                "height": img.height,
                "aspect_ratio": round(img.aspect_ratio, 2),
                "megapixels": round(img.megapixels, 2),
                "storage_profile": {
                    "id": str(img.storage_profile.id),
                    "name": img.storage_profile.name,
                    "backend": img.storage_profile.backend
                },
                "storage_key": img.storage_key,
                "checksum": img.checksum,
                "source_of_origin": img.source_of_origin,
                "tags": [tag.name for tag in img.tags.all()],
                "uploaded_by": img.uploaded_by.username if img.uploaded_by else None,
                "created_at": img.created_at.isoformat(),
                "updated_at": img.updated_at.isoformat(),
                "download_url": await get_dowload_url(img, expiration=3600)
            }
            for img in result["images"]
        ]
    }

@router.get(
    "/{image_id}",
    summary="Get Image",
    description="Get detailed information about a specific image"
)
async def get_image(
    image_id: UUID,
    ctx: RequestContext = Depends(get_request_context)
):
    """Get specific image details."""
    
    try:
        image = Image.objects.select_related(
            'storage_profile',
            'uploaded_by',
            'project'
        ).prefetch_related('tags').get(
            id=image_id,
            organization=ctx.organization
        )
        
        return {
            "id": str(image.id),
            "name": image.name,
            "original_filename": image.original_filename,
            "file_format": image.file_format,
            "file_size": image.file_size,
            "file_size_mb": round(image.file_size_mb, 2),
            "width": image.width,
            "height": image.height,
            "aspect_ratio": round(image.aspect_ratio, 2),
            "megapixels": round(image.megapixels, 2),
            "checksum": image.checksum,
            "annotated": image.annotated,
            "processed": image.processed,
            "source_of_origin": image.source_of_origin,
            "storage_profile": {
                "id": str(image.storage_profile.id),
                "name": image.storage_profile.name,
                "backend": image.storage_profile.backend
            },
            "storage_key": image.storage_key,
            "tags": [
                {
                    "name": tag.name,
                    "color": tag.color
                }
                for tag in image.tags.all()
            ],
            "meta_info": image.meta_info,
            "uploaded_by": image.uploaded_by.username if image.uploaded_by else None,
            "created_at": image.created_at.isoformat(),
            "updated_at": image.updated_at.isoformat(),
            "download_url": image.get_download_url(expiration=3600)
        }
        
    except Image.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )


@router.get(
    "/{image_id}/download",
    summary="Download Image",
    description="Download the actual image file"
)
async def download_image(
    image_id: UUID,
    ctx: RequestContext = Depends(get_request_context)
):
    """Download image file."""
    
    try:
        image = Image.objects.get(
            id=image_id,
            organization=ctx.organization
        )
        
        # Get storage adapter
        adapter = get_storage_adapter_for_profile(image.storage_profile)
        
        # Download file
        file_data = adapter.download_file(image.storage_key)
        
        # Return as streaming response
        return StreamingResponse(
            BytesIO(file_data),
            media_type=f"image/{image.file_format}",
            headers={
                "Content-Disposition": f'attachment; filename="{image.original_filename}"'
            }
        )
        
    except Image.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )


@router.delete(
    "/{image_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Image",
    description="Delete image from storage and database"
)
async def delete_image(
    image_id: UUID,
    ctx: RequestContext = Depends(get_request_context)
):
    """Delete image."""
    
    # Require admin role for deletion
    ctx.require_role('admin', 'owner')
    
    try:
        image = Image.objects.get(
            id=image_id,
            organization=ctx.organization
        )
        
        # Delete from storage first
        try:
            image.delete_from_storage()
            logger.info(f"Deleted file from storage: {image.storage_key}")
        except Exception as e:
            logger.error(f"Failed to delete file from storage: {e}")
            # Continue with database deletion even if storage deletion fails
        
        # Delete from database
        image.delete()
        logger.info(f"Deleted image record: {image_id}")
        
    except Image.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )


# ============= Image Tagging Endpoints =============

@router.post(
    "/{image_id}/tags",
    status_code=status.HTTP_201_CREATED,
    summary="Add Tags to Image",
    description="Add one or more tags to an image"
)
async def add_tags_to_image(
    image_id: UUID,
    tag_names: List[str],
    ctx: RequestContext = Depends(get_request_context)
):
    """Add tags to an image."""
    
    try:
        image = Image.objects.get(
            id=image_id,
            organization=ctx.organization
        )
        
        added_tags = []
        
        for tag_name in tag_names:
            # Get or create tag
            tag, created = Tag.objects.get_or_create(
                organization=ctx.organization,
                name=tag_name.strip()
            )
            
            # Create image-tag association (if not exists)
            image_tag, created = ImageTag.objects.get_or_create(
                image=image,
                tag=tag,
                defaults={'tagged_by': ctx.user}
            )
            
            if created:
                added_tags.append(tag_name)
        
        return {
            "message": f"Added {len(added_tags)} tags",
            "added_tags": added_tags,
            "total_tags": image.tags.count()
        }
        
    except Image.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )


@router.delete(
    "/{image_id}/tags/{tag_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove Tag from Image",
    description="Remove a specific tag from an image"
)
async def remove_tag_from_image(
    image_id: UUID,
    tag_name: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """Remove a tag from an image."""
    
    try:
        image = Image.objects.get(
            id=image_id,
            organization=ctx.organization
        )
        
        tag = Tag.objects.get(
            organization=ctx.organization,
            name=tag_name
        )
        
        ImageTag.objects.filter(
            image=image,
            tag=tag
        ).delete()
        
        logger.info(f"Removed tag '{tag_name}' from image {image_id}")
        
    except Image.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )
    except Tag.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tag '{tag_name}' not found"
        )


# ============= Image Project Association Endpoints =============

@router.post(
    "/{image_id}/projects/{project_id}",
    status_code=status.HTTP_201_CREATED,
    summary="Associate Image with Project",
    description="Add an image to a project"
)
async def associate_image_with_project(
    image_id: UUID,
    project_id: UUID,
    role: Optional[str] = Query(None, description="Role in project (training, validation, test)"),
    ctx: RequestContext = Depends(get_request_context)
):
    """Associate an image with a project."""
    
    from projects.models import Project
    
    try:
        image = Image.objects.get(
            id=image_id,
            organization=ctx.organization
        )
        
        project = Project.objects.get(
            id=project_id,
            organization=ctx.organization
        )
        
        # Create or update association
        association, created = ImageProject.objects.get_or_create(
            image=image,
            project=project,
            defaults={
                'added_by': ctx.user,
                'role': role
            }
        )
        
        if not created and role:
            association.role = role
            association.save()
        
        return {
            "message": "Image associated with project" if created else "Association updated",
            "image_id": str(image_id),
            "project_id": str(project_id),
            "role": role,
            "created": created
        }
        
    except Image.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )
    except Project.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )


@router.delete(
    "/{image_id}/projects/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Disassociate Image from Project",
    description="Remove an image from a project"
)
async def disassociate_image_from_project(
    image_id: UUID,
    project_id: UUID,
    ctx: RequestContext = Depends(get_request_context)
):
    """Remove image from project."""
    
    try:
        image = Image.objects.get(
            id=image_id,
            organization=ctx.organization
        )
        
        ImageProject.objects.filter(
            image=image,
            project_id=project_id
        ).delete()
        
        logger.info(f"Disassociated image {image_id} from project {project_id}")
        
    except Image.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )


# ============= Statistics Endpoints =============

@router.get(
    "/statistics/overview",
    summary="Get Image Statistics",
    description="Get overview statistics for organization's images"
)
async def get_image_statistics(
    ctx: RequestContext = Depends(get_request_context),
    project_id: Optional[UUID] = Query(None)
):
    """Get image statistics."""
    
    from django.db.models import Count, Sum, Avg
    
    queryset = Image.objects.filter(organization=ctx.organization)
    
    if project_id:
        queryset = queryset.filter(project_associations__project_id=project_id)
    
    # Calculate statistics
    stats = queryset.aggregate(
        total_images=Count('id'),
        total_size=Sum('file_size'),
        avg_width=Avg('width'),
        avg_height=Avg('height'),
        annotated_count=Count('id', filter=models.Q(annotated=True)),
        processed_count=Count('id', filter=models.Q(processed=True))
    )
    
    # Format by backend
    storage_stats = []
    for profile in StorageProfile.objects.filter(organization=ctx.organization):
        count = Image.objects.filter(
            organization=ctx.organization,
            storage_profile=profile
        ).count()
        
        if count > 0:
            storage_stats.append({
                "storage_profile": profile.name,
                "backend": profile.backend,
                "image_count": count
            })
    
    # Format by file format
    format_stats = Image.objects.filter(
        organization=ctx.organization
    ).values('file_format').annotate(
        count=Count('id')
    ).order_by('-count')
    
    return {
        "total_images": stats['total_images'] or 0,
        "total_size_bytes": stats['total_size'] or 0,
        "total_size_gb": round((stats['total_size'] or 0) / (1024**3), 2),
        "annotated_images": stats['annotated_count'] or 0,
        "processed_images": stats['processed_count'] or 0,
        "average_dimensions": {
            "width": round(stats['avg_width'] or 0),
            "height": round(stats['avg_height'] or 0)
        },
        "by_storage": storage_stats,
        "by_format": list(format_stats)
    }