from __future__ import annotations
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class Image(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    image_id: Optional[str] = None
    name: Optional[str] = None
    original_filename: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    file_format: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    source_of_origin: Optional[str] = None
    storage_key: Optional[str] = None
    download_url: Optional[str] = None
    tags: List[str] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ImageUploadResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    duplicate: bool = False
    image: Optional[dict] = None
    download_url: Optional[str] = None

class ProjectImage(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    image_id: Optional[str] = None
    name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    file_format: Optional[str] = None
    download_url: Optional[str] = None
    status: Optional[str] = None
    split: Optional[str] = None
    annotated: Optional[bool] = None
    reviewed: Optional[bool] = None
    tags: List[str] = []
    annotations: list = []

class ProjectImagesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    total: int = 0
    annotated: int = 0
    unannotated: int = 0
    reviewed: int = 0
    images: List[ProjectImage] = []
    image_ids: Optional[list] = None
