"""
FastAPI routes for image management and upload.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import Response, StreamingResponse
import logging
import mimetypes
import re
import time
from pathlib import Path
from django.contrib.auth import get_user_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/media")


def _media_url(storage_backend: str, storage_key: str) -> str:
    """Return a browser-accessible URL for a stored file.

    Cloud backends (Azure, S3, MinIO) already return HTTP presigned URLs.
    Local storage serves files through the built-in /api/v1/media/files/
    endpoint via a relative URL, which works same-origin behind any proxy
    (nginx in production, the Vite dev proxy in development).
    """
    if storage_backend == "local":
        from storage.services import get_local_media_url
        return get_local_media_url(storage_key)
    return ""  # caller will call get_download_url for cloud backends


# Files larger than this are streamed in ranged reads of this size, so the
# proxy never holds a whole large image in memory.
PROXY_CHUNK_BYTES = 4 * 1024 * 1024


def _cache_headers(exp: str, etag: str) -> dict:
    """Let the browser keep the file for as long as its signed URL is valid
    (the URL is stable within a window, see storage.signing.signed_query)."""
    max_age = max(0, int(exp) - int(time.time())) if exp and exp.isdigit() else 0
    return {"Cache-Control": f"private, max-age={max_age}", "ETag": etag}


def _not_modified(request: Request, etag: str) -> bool:
    return etag in [t.strip() for t in request.headers.get("If-None-Match", "").split(",")]


@router.get("/images/{image_id}", include_in_schema=True)
async def serve_image_via_proxy(request: Request, image_id: str, exp: str = None, sig: str = None):
    """Stream an image whose storage cannot mint short-lived URLs (e.g. an
    external Azure profile with a SAS-only credential), authorized by the
    signed query string from Image.get_proxy_url — the storage token never
    reaches the browser.

    Browsers revalidating a cached copy (If-None-Match) get a 304 without
    the file being read from storage."""
    from asgiref.sync import sync_to_async
    from storage.signing import verify

    if not verify(f"/api/v1/media/images/{image_id}", exp, sig):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Invalid or expired link")

    @sync_to_async
    def lookup():
        from images.models import Image

        image = Image.objects.select_related("storage_profile").filter(image_id=image_id).first()
        if image is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Image not found")
        return image

    image = await lookup()
    version = image.checksum or (int(image.updated_at.timestamp()) if image.updated_at else image.file_size)
    etag = f'"{image.image_id}-{version}"'
    headers = _cache_headers(exp, etag)
    if _not_modified(request, etag):
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)

    @sync_to_async
    def first_read():
        from perceptra_storage import StorageNotFoundError
        from storage.services import get_storage_adapter_for_profile

        adapter = get_storage_adapter_for_profile(image.storage_profile)
        try:
            if (image.file_size or 0) > PROXY_CHUNK_BYTES:
                return adapter, adapter.download_range(image.storage_key, 0, PROXY_CHUNK_BYTES), True
            return adapter, adapter.download_file(image.storage_key), False
        except StorageNotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="File not found in storage")

    # Read the first part before answering, so a missing file is still a 404.
    adapter, first, chunked = await first_read()
    content_type = mimetypes.guess_type(image.storage_key)[0] or f"image/{image.file_format or 'jpeg'}"
    if not chunked:
        return Response(content=first, media_type=content_type, headers=headers)

    def rest():
        yield first
        offset, chunk = len(first), first
        while len(chunk) == PROXY_CHUNK_BYTES:
            try:
                chunk = adapter.download_range(image.storage_key, offset, PROXY_CHUNK_BYTES)
            except Exception:
                # A size that is an exact multiple of the chunk ends with an
                # out-of-range read on some backends; anything else is logged.
                if offset < (image.file_size or 0):
                    logger.warning("Proxy stream of image %s stopped at byte %d", image.image_id, offset,
                                   exc_info=True)
                break
            if not chunk:
                break
            offset += len(chunk)
            yield chunk

    return StreamingResponse(rest(), media_type=content_type, headers=headers)


@router.get("/files/{storage_path:path}", include_in_schema=True)
async def serve_local_file(request: Request, storage_path, exp: str = None, sig: str = None):
    """Stream a file from local storage with HTTP range request support.

    Range support is required for HTML5 video seeking (browsers send
    'Range: bytes=N-M' when the user scrubs the timeline).

    With MEDIA_REQUIRE_SIGNED_URLS the request must carry the signature that
    storage.services.get_local_media_url adds (it always adds one).
    """

    from django.conf import settings

    if getattr(settings, "MEDIA_REQUIRE_SIGNED_URLS", False):
        from storage.signing import verify

        if not verify(f"/api/v1/media/files/{storage_path.lstrip('/')}", exp, sig):
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Invalid or expired link")

    # Stored keys are absolute filesystem paths, but the leading slash may be
    # lost in transit (nginx merges "//" into "/"), so root the path ourselves.
    # resolve() normalizes ".." segments and symlinks before the allowlist
    # check, so the endpoint can only ever serve files under MEDIA_SERVE_ROOTS.
    full_path = (Path("/") / storage_path.lstrip("/")).resolve()
    allowed_roots = [Path(root).resolve() for root in settings.MEDIA_SERVE_ROOTS]
    if not any(full_path.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="File not found")

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="File not found")

    content_type, _ = mimetypes.guess_type(str(full_path))
    content_type = content_type or "application/octet-stream"
    stat_result = full_path.stat()
    file_size = stat_result.st_size
    cache_headers = _cache_headers(exp, f'"{stat_result.st_mtime_ns:x}-{file_size:x}"')
    if _not_modified(request, cache_headers["ETag"]):
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=cache_headers)

    range_header = request.headers.get("Range")
    if range_header:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            def iter_range():
                with open(full_path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            return StreamingResponse(
                iter_range(),
                status_code=206,
                media_type=content_type,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(length),
                    **cache_headers,
                },
            )

    def iter_file():
        with open(full_path, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            **cache_headers,
        },
    )