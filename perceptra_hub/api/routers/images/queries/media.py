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
    Local storage serves files through the built-in /api/v1/media/files/ endpoint.

    In production (same-origin, behind nginx) the URL is relative so nginx can
    proxy it to the backend transparently.

    In dev Docker the Vite proxy runs inside its container and cannot reach the
    backend via localhost. Setting MEDIA_BASE_URL=http://localhost:29085 on the
    backend makes the URL absolute so the browser hits the backend port directly.
    """
    if storage_backend == "local":
        from django.conf import settings
        base = getattr(settings, 'MEDIA_BASE_URL', '')
        return f"{base}/api/v1/media/files/{storage_key}"
    return ""  # caller will call get_download_url for cloud backends


@router.get("/files/{storage_path:path}", include_in_schema=True)
async def serve_local_file(request: Request, storage_path):
    """Stream a file from local storage with HTTP range request support.

    Range support is required for HTML5 video seeking (browsers send
    'Range: bytes=N-M' when the user scrubs the timeline).
    """
    
    full_path = Path(storage_path)

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