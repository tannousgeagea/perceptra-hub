"""
FastAPI routes for image management and upload.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
import logging
import mimetypes
import re
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


@router.get("/files/{storage_path:path}", include_in_schema=True)
async def serve_local_file(request: Request, storage_path):
    """Stream a file from local storage with HTTP range request support.

    Range support is required for HTML5 video seeking (browsers send
    'Range: bytes=N-M' when the user scrubs the timeline).
    """
    
    from django.conf import settings

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
    file_size = full_path.stat().st_size

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
        },
    )