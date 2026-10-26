"""
Folder-style browsing of a storage profile — the file picker of "Register from
storage" in the console.

    GET /api/v1/storage/profiles/{storage_profile_id}/browse?prefix=&token=&page_size=&images_only=

One page of sub-folders and files under `prefix` (native paginated listing of
the backend, perceptra_storage list_page), each file flagged when it is
already registered in Perceptra. Needs list permission on the storage: a
credential without it (e.g. an Azure SAS with sp=r) gets a 403 telling the
user to paste keys or upload a manifest instead.
"""
import logging
import os
from typing import List, Optional
from uuid import UUID

from asgiref.sync import sync_to_async
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from api.dependencies import RequestContext, require_permission
from perceptra_storage import StorageError, StoragePermissionError
from storage.models import StorageProfile
from storage.services import StorageServiceError, get_storage_adapter_for_profile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/storage", tags=["Storage Management"])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class BrowseFile(BaseModel):
    key: str
    name: str
    size: int
    last_modified: Optional[str] = None
    content_type: Optional[str] = None
    registered: bool = False
    image_id: Optional[str] = None


class BrowseResponse(BaseModel):
    storage_profile_id: str
    prefix: str
    folders: List[str]
    files: List[BrowseFile]
    next_token: Optional[str] = None


def _browse(org, storage_profile_id: UUID, prefix: str, token: Optional[str], page_size: int,
            images_only: bool) -> BrowseResponse:
    from images.models import Image

    profile = StorageProfile.objects.filter(
        organization=org, storage_profile_id=storage_profile_id, is_active=True,
    ).first()
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Storage profile not found")

    try:
        page = get_storage_adapter_for_profile(profile).list_page(
            prefix=prefix, delimiter="/", page_size=page_size, continuation_token=token or None,
        )
    except StoragePermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This storage credential cannot list files (e.g. an Azure SAS without 'l'). "
                   "Paste the file keys or upload a manifest instead, or rotate to a credential with list permission.",
        )
    except (StorageError, StorageServiceError) as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Listing failed: {e}")

    objects = [
        o for o in page.objects
        if not images_only or os.path.splitext(o.key)[1].lower() in IMAGE_EXTENSIONS
    ]
    registered = dict(
        Image.objects.filter(storage_profile=profile, storage_key__in=[o.key for o in objects])
        .values_list("storage_key", "image_id")
    )
    return BrowseResponse(
        storage_profile_id=str(profile.storage_profile_id),
        prefix=prefix,
        folders=page.prefixes,
        files=[
            BrowseFile(
                key=o.key,
                name=o.key.rsplit("/", 1)[-1],
                size=o.size or 0,
                last_modified=o.last_modified.isoformat() if o.last_modified else None,
                content_type=o.content_type,
                registered=o.key in registered,
                image_id=registered.get(o.key),
            )
            for o in objects
        ],
        next_token=page.next_token,
    )


@router.get("/profiles/{storage_profile_id}/browse", response_model=BrowseResponse,
            summary="Browse a storage profile (folders + files, paginated)")
async def browse_storage_profile(
    storage_profile_id: UUID,
    prefix: str = Query("", description="Folder to list, e.g. 'images/2026/10/'"),
    token: Optional[str] = Query(None, description="next_token of the previous page"),
    page_size: int = Query(200, ge=1, le=1000),
    images_only: bool = Query(True, description="Only list image files"),
    ctx: RequestContext = Depends(require_permission("write")),
):
    return await sync_to_async(_browse)(ctx.organization, storage_profile_id, prefix, token, page_size, images_only)
