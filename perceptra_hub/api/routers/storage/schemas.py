"""
Pydantic schemas for storage management API.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum
from uuid import UUID


class StorageBackendEnum(str, Enum):
    """Supported storage backend types."""
    AZURE = "azure"
    S3 = "s3"
    MINIO = "minio"
    LOCAL = "local"


class SecretProviderEnum(str, Enum):
    """Secret management provider types."""
    VAULT = "vault"
    AZURE_KV = "azure_kv"
    AWS_SM = "aws_sm"
    LOCAL_ENC = "local_enc"


# ============= Secret Reference Schemas =============

class SecretRefCreate(BaseModel):
    """Schema for creating a secret reference."""
    provider: SecretProviderEnum
    path: str = Field(..., min_length=1, max_length=500)
    key: str = Field(..., min_length=1, max_length=100)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "vault",
                "path": "secret/data/storage/s3-prod",
                "key": "credentials",
                "metadata": {"environment": "production"}
            }
        }


class SecretRefUpdate(BaseModel):
    """Schema for updating a secret reference."""
    provider: Optional[SecretProviderEnum] = None
    path: Optional[str] = Field(None, min_length=1, max_length=500)
    key: Optional[str] = Field(None, min_length=1, max_length=100)
    metadata: Optional[Dict[str, Any]] = None


class SecretRefResponse(BaseModel):
    """Schema for secret reference response."""
    id: int
    credential_ref_id: UUID
    provider: str
    path: str
    key: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============= Storage Profile Schemas =============

class StorageProfileCreate(BaseModel):
    """Schema for creating a storage profile."""
    name: str = Field(..., min_length=1, max_length=100)
    backend: StorageBackendEnum
    region: Optional[str] = Field(None, max_length=50)
    is_default: bool = False
    config: Dict[str, Any] = Field(...)
    credential_ref_id: Optional[str] = None

    @validator('config')
    def validate_config(cls, v, values):
        """Validate config based on backend type."""
        backend = values.get('backend')
        
        if backend == StorageBackendEnum.S3:
            if 'bucket_name' not in v:
                raise ValueError("S3 backend requires 'bucket_name' in config")
        
        elif backend == StorageBackendEnum.AZURE:
            if 'container_name' not in v or 'account_name' not in v:
                raise ValueError(
                    "Azure backend requires 'container_name' and 'account_name' in config"
                )
        
        elif backend == StorageBackendEnum.MINIO:
            if 'bucket_name' not in v or 'endpoint_url' not in v:
                raise ValueError(
                    "MinIO backend requires 'bucket_name' and 'endpoint_url' in config"
                )
        
        elif backend == StorageBackendEnum.LOCAL:
            if 'base_path' not in v:
                raise ValueError("Local backend requires 'base_path' in config")
        
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Production S3",
                "backend": "s3",
                "region": "us-west-2",
                "is_default": True,
                "config": {
                    "bucket_name": "my-company-storage",
                    "region": "us-west-2"
                },
                "credential_ref_id": "uuid-here"
            }
        }


class StorageProfileUpdate(BaseModel):
    """Schema for updating a storage profile."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    region: Optional[str] = Field(None, max_length=50)
    is_default: Optional[bool] = None
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    credential_ref_id: Optional[str] = None


class StorageProfileResponse(BaseModel):
    """Schema for storage profile response."""
    id: int
    storage_profile_id: UUID
    name: str
    backend: str
    region: Optional[str]
    is_default: bool
    is_active: bool
    config: Dict[str, Any]
    credential_ref_id: Optional[int]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class StorageProfileListResponse(BaseModel):
    """Schema for paginated storage profile list."""
    total: int
    page: int
    page_size: int
    profiles: List[StorageProfileResponse]


# ============= Storage Operations Schemas =============

class ConnectionTestRequest(BaseModel):
    """Schema for testing storage connection."""
    timeout: int = Field(default=10, ge=1, le=60)


class ConnectionTestResponse(BaseModel):
    """Schema for connection test response."""
    success: bool
    message: str
    error: Optional[str] = None


class FileUploadRequest(BaseModel):
    """Schema for file upload metadata."""
    key: str = Field(..., description="Destination path/key in storage")
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


class FileUploadResponse(BaseModel):
    """Schema for file upload response."""
    key: str
    size: int
    content_type: Optional[str]
    uploaded_at: datetime


class FileDownloadRequest(BaseModel):
    """Schema for file download request."""
    key: str = Field(..., description="File path/key in storage")


class FileListRequest(BaseModel):
    """Schema for listing files."""
    prefix: str = Field(default="", description="Filter by prefix")
    max_results: int = Field(default=100, ge=1, le=1000)


class FileMetadata(BaseModel):
    """Schema for file metadata."""
    key: str
    size: int
    last_modified: datetime
    etag: Optional[str]
    content_type: Optional[str]


class FileListResponse(BaseModel):
    """Schema for file list response."""
    files: List[FileMetadata]
    count: int
    prefix: str


class PresignedUrlRequest(BaseModel):
    """Schema for presigned URL generation."""
    key: str = Field(..., description="File path/key")
    expiration: int = Field(default=3600, ge=60, le=86400, description="Expiration in seconds")
    method: str = Field(default="GET", pattern="^(GET|PUT|DELETE)$")


class PresignedUrlResponse(BaseModel):
    """Schema for presigned URL response."""
    url: str
    expires_at: datetime
    method: str


class FileDeleteRequest(BaseModel):
    """Schema for file deletion."""
    key: str = Field(..., description="File path/key to delete")


class FileDeleteResponse(BaseModel):
    """Schema for file deletion response."""
    success: bool
    key: str
    message: str


# ============= Batch Operations Schemas =============

class BatchUploadItem(BaseModel):
    """Schema for batch upload item."""
    key: str
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


class BatchUploadRequest(BaseModel):
    """Schema for batch upload request."""
    files: List[BatchUploadItem] = Field(..., max_items=100)


class BatchUploadResult(BaseModel):
    """Schema for individual batch upload result."""
    key: str
    success: bool
    size: Optional[int] = None
    error: Optional[str] = None


class BatchUploadResponse(BaseModel):
    """Schema for batch upload response."""
    total: int
    successful: int
    failed: int
    results: List[BatchUploadResult]


class BatchDeleteRequest(BaseModel):
    """Schema for batch delete request."""
    keys: List[str] = Field(..., max_items=100)


class BatchDeleteResult(BaseModel):
    """Schema for individual batch delete result."""
    key: str
    success: bool
    error: Optional[str] = None


class BatchDeleteResponse(BaseModel):
    """Schema for batch delete response."""
    total: int
    successful: int
    failed: int
    results: List[BatchDeleteResult]


# ============= Error Schemas =============

class ErrorDetail(BaseModel):
    """Schema for error details."""
    field: Optional[str] = None
    message: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str
    details: Optional[List[ErrorDetail]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Validation error",
                "details": [
                    {
                        "field": "config.bucket_name",
                        "message": "Field required",
                        "code": "required"
                    }
                ],
                "timestamp": "2025-10-20T10:30:00Z"
            }
        }


# ============= Statistics Schemas =============

class StorageStatistics(BaseModel):
    """Schema for storage statistics."""
    profile_id: str
    profile_name: str
    backend: str
    total_files: int
    total_size: int
    last_upload: Optional[datetime]


class TenantStorageStatistics(BaseModel):
    """Schema for tenant-wide storage statistics."""
    total_profiles: int
    active_profiles: int
    total_files: int
    total_size: int
    profiles: List[StorageStatistics]