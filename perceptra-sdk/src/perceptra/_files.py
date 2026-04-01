"""
File upload normalization for image uploads.

Accepts str/Path, bytes, or file-like objects and normalizes to
the httpx multipart format: (filename, data, content_type).
"""
from __future__ import annotations

import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Optional, Tuple, Union

FileInput = Union[str, Path, bytes, BinaryIO]


def prepare_file(
    file: FileInput,
    *,
    name: Optional[str] = None,
) -> Tuple[str, Any, str]:
    """
    Normalize a file input into an httpx-compatible tuple.

    Returns:
        (filename, file_data, content_type)
    """
    if isinstance(file, (str, Path)):
        path = Path(file)
        filename = name or path.name
        content_type = _guess_content_type(filename)
        return (filename, open(path, "rb"), content_type)

    if isinstance(file, bytes):
        filename = name or "upload"
        content_type = _guess_content_type(filename)
        return (filename, BytesIO(file), content_type)

    # file-like object
    filename = name or getattr(file, "name", "upload")
    if isinstance(filename, str) and "/" in filename:
        filename = filename.rsplit("/", 1)[-1]
    content_type = _guess_content_type(filename)
    return (filename, file, content_type)


def _guess_content_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"
