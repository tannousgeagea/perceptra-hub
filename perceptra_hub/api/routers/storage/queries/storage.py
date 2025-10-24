"""
FastAPI routes for storage management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List
from uuid import UUID
from asgiref.sync import sync_to_async
 
import logging

from api.routers.storage.schemas import (
    # Storage Profile schemas
    StorageProfileCreate,
    StorageProfileUpdate,
    StorageProfileResponse,
    StorageProfileListResponse,
    # Secret Reference schemas
    SecretRefCreate,
    SecretRefUpdate,
    SecretRefResponse,
    # Operation schemas
    ConnectionTestRequest,
    ConnectionTestResponse,
    FileUploadResponse,
    FileListRequest,
    FileListResponse,
    PresignedUrlRequest,
    PresignedUrlResponse,
    FileDeleteRequest,
    FileDeleteResponse,
    BatchUploadRequest,
    BatchUploadResponse,
    BatchDeleteRequest,
    BatchDeleteResponse,
    StorageStatistics,
    ErrorResponse,
)
from api.dependencies import get_current_organization, get_db
from storage.models import StorageProfile, SecretRef
from organizations.models import Organization
from storage.services import (
    get_storage_adapter_for_profile,
    test_storage_profile_connection,
    get_default_storage_adapter,
    StorageServiceError,
)
from perceptra_storage import (
    StorageError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageConnectionError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/storage", tags=["Storage Management"])


# ============= Secret Reference Endpoints =============
@sync_to_async
def create_secret_reference(organization: Organization, secret_data: SecretRefCreate) -> SecretRefCreate:
    """
    Synchronous function to create a secret reference.
    """
    secret_ref = SecretRef.objects.create(
        organization=organization,
        provider=secret_data.provider,
        path=secret_data.path,
        key=secret_data.key,
        metadata=secret_data.metadata
    )
    
    logger.info(f"Created secret reference {secret_ref.id} for organization {organization.id}")
    return SecretRefResponse.from_orm(secret_ref)

@router.post(
    "/secrets",
    response_model=SecretRefResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Secret Reference",
    description="Create a reference to externally stored credentials"
)
async def create_secret_ref(
    secret_data: SecretRefCreate,
    organization=Depends(get_current_organization),
    db=Depends(get_db)
):
    """Create a new secret reference for the organization."""
    try:
        return await create_secret_reference(organization, secret_data)
        
    except Exception as e:
        logger.error(f"Failed to create secret reference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create secret reference: {str(e)}"
        )


@router.get(
    "/secrets",
    response_model=List[SecretRefResponse],
    summary="List Secret References",
    description="Get all secret references for the current organization"
)
async def list_secret_refs(
    organization=Depends(get_current_organization),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """List all secret references for the organization."""
    @sync_to_async
    def fetch_secret_refs(organization: Organization, skip:int, limit:int):
        secrets = SecretRef.objects.filter(organization=organization)[skip:skip + limit]
        return [SecretRefResponse.from_orm(s) for s in secrets]
    
    return await fetch_secret_refs(organization=organization, skip=skip, limit=limit)


@router.get(
    "/secrets/{secret_id}",
    response_model=SecretRefResponse,
    summary="Get Secret Reference",
    description="Get a specific secret reference by ID"
)
async def get_secret_ref(
    secret_id: UUID,
    organization=Depends(get_current_organization)
):
    """Get a specific secret reference."""
    
    @sync_to_async
    def fetch_secret_ref(secret_id:UUID, organization:Organization):
        try:
            secret = SecretRef.objects.get(credential_ref_id=secret_id, organization=organization)
            return SecretRefResponse.from_orm(secret)
        except SecretRef.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Secret reference {secret_id} not found"
        )
            
    return await fetch_secret_ref(secret_id, organization)


@router.delete(
    "/secrets/{secret_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Secret Reference",
    description="Delete a secret reference (only if not in use)"
)
async def delete_secret_ref(
    secret_id: UUID,
    organization=Depends(get_current_organization)
):
    """Delete a secret reference."""
    @sync_to_async
    def delete_secret_reference(secret_id: UUID, organization: Organization):
        """
        Synchronous function to delete a secret reference.
        """
        try:
            secret = SecretRef.objects.get(credential_ref_id=secret_id, organization=organization)
            
            # Check if in use
            if secret.storage_profiles.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete secret reference that is in use by storage profiles"
                )
            
            secret.delete()
            
        except SecretRef.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Secret reference {secret_id} not found"
            )
            
    await delete_secret_reference(secret_id, organization)
    logger.info(f"Deleted secret reference {secret_id}")


# ============= Storage Profile Endpoints =============
@sync_to_async
def create_storage_profile_record(organization: Organization, profile_data: StorageProfileCreate):
    """
    Synchronous function to create a storage profile.
    """
    # Get credential reference if provided
    credential_ref = None
    if profile_data.credential_ref_id:
        try:
            credential_ref = SecretRef.objects.get(
                credential_ref_id=profile_data.credential_ref_id,
                organization=organization
            )
        except SecretRef.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Secret reference {profile_data.credential_ref_id} not found"
            )
    
    # Create profile
    profile = StorageProfile(
        organization=organization,
        name=profile_data.name,
        backend=profile_data.backend,
        region=profile_data.region,
        is_default=profile_data.is_default,
        config=profile_data.config,
        credential_ref=credential_ref
    )
    
    # Test connection before saving
    success, error = test_storage_profile_connection(profile)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Storage connection test failed: {error}"
        )
    
    # Save profile
    profile.save()
    
    return profile

@router.post(
    "/profiles",
    response_model=StorageProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Storage Profile",
    description="Create a new storage profile for the organization"
)
async def create_storage_profile(
    profile_data: StorageProfileCreate,
    organization=Depends(get_current_organization)
):
    """Create a new storage profile."""
    try:
        # Get credential reference if provided
        profile = await create_storage_profile_record(organization, profile_data)
        logger.info(f"Created storage profile {profile.id} for organization {organization.id}")
        return StorageProfileResponse.from_orm(profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create storage profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create storage profile: {str(e)}"
        )


@router.get(
    "/profiles",
    response_model=StorageProfileListResponse,
    summary="List Storage Profiles",
    description="Get all storage profiles for the current organization"
)
async def list_storage_profiles(
    organization=Depends(get_current_organization),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    backend: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    is_default: Optional[bool] = Query(None)
):
    """List storage profiles with optional filtering."""
    @sync_to_async
    def get_storage_profiles_list(
        organization: Organization,
        skip: int,
        limit: int,
        backend: Optional[str],
        is_active: Optional[bool],
        is_default: Optional[bool]
    ):
        """
        Synchronous function to get storage profiles list.
        """
        queryset = StorageProfile.objects.filter(organization=organization)
        
        if backend:
            queryset = queryset.filter(backend=backend)
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active)
        if is_default is not None:
            queryset = queryset.filter(is_default=is_default)
        
        total = queryset.count()
        profiles = list(queryset[skip:skip + limit])
        
        return {
            "total": total,
            "profiles": profiles,
            "skip": skip,
            "limit": limit
        }
        
    result = await get_storage_profiles_list(
        organization, skip, limit, backend, is_active, is_default
    )

    return StorageProfileListResponse(
        total=result["total"],
        page=result["skip"] // result["limit"] + 1,
        page_size=result["limit"],
        profiles=[StorageProfileResponse.from_orm(p) for p in result["profiles"]]
    )


@router.get(
    "/profiles/{profile_id}",
    response_model=StorageProfileResponse,
    summary="Get Storage Profile",
    description="Get a specific storage profile by ID"
)
async def get_storage_profile(
    profile_id: UUID,
    organization=Depends(get_current_organization)
):
    """Get a specific storage profile."""
    @sync_to_async
    def fetch_storage_profile(profile_id:UUID, organization: Organization):
        try:
            profile = StorageProfile.objects.get(storage_profile_id=profile_id, organization=organization)
            return StorageProfileResponse.from_orm(profile)
        except StorageProfile.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Storage profile {profile_id} not found"
            )
    return await fetch_storage_profile(profile_id, organization)


@router.put(
    "/profiles/{profile_id}",
    response_model=StorageProfileResponse,
    summary="Update Storage Profile",
    description="Update an existing storage profile"
)
async def update_storage_profile(
    profile_id: UUID,
    profile_data: StorageProfileUpdate,
    organization=Depends(get_current_organization)
):
    """Update a storage profile."""
    @sync_to_async
    def update_storage_profile_sync(profile_id:UUID, organization=organization):
        try:
            profile = StorageProfile.objects.get(storage_profile_id=profile_id, organization=organization)
            
            # Update fields
            if profile_data.name is not None:
                profile.name = profile_data.name
            if profile_data.region is not None:
                profile.region = profile_data.region
            if profile_data.is_default is not None:
                profile.is_default = profile_data.is_default
            if profile_data.is_active is not None:
                profile.is_active = profile_data.is_active
            if profile_data.config is not None:
                profile.config = profile_data.config
            if profile_data.credential_ref_id is not None:
                credential_ref = SecretRef.objects.get(
                    credential_ref_id=profile_data.credential_ref_id,
                    organization=organization
                )
                profile.credential_ref = credential_ref
            
            profile.save()
            
            logger.info(f"Updated storage profile {profile_id}")
            return StorageProfileResponse.from_orm(profile)
            
        except StorageProfile.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Storage profile {profile_id} not found"
            )
        except SecretRef.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Secret reference not found"
            )
    
    return await update_storage_profile_sync(profile_id, organization)


@router.delete(
    "/profiles/{profile_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Storage Profile",
    description="Delete a storage profile"
)
async def delete_storage_profile(
    profile_id: UUID,
    organization=Depends(get_current_organization)
):
    """Delete a storage profile."""
    @sync_to_async
    def delete_storage_profile_sync(profile_id:UUID, organization:Organization):
        try:
            profile = StorageProfile.objects.get(storage_profile_id=profile_id, organization=organization)
            profile.delete()
            logger.info(f"Deleted storage profile {profile_id}")
            
        except StorageProfile.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Storage profile {profile_id} not found"
            )

    await delete_storage_profile_sync(profile_id, organization)

# ============= Storage Operations Endpoints =============

@router.post(
    "/profiles/{profile_id}/test-connection",
    response_model=ConnectionTestResponse,
    summary="Test Storage Connection",
    description="Test connection to a storage profile"
)
async def test_connection(
    profile_id: UUID,
    request: ConnectionTestRequest,
    organization=Depends(get_current_organization)
):
    """Test connection to a storage profile."""
    @sync_to_async
    def test_connection_sync(profile_id:UUID, organization:Organization):
        try:
            profile = StorageProfile.objects.get(storage_profile_id=profile_id, organization=organization)
            
            success, error = test_storage_profile_connection(profile)
            
            if success:
                return ConnectionTestResponse(
                    success=True,
                    message="Connection successful"
                )
            else:
                return ConnectionTestResponse(
                    success=False,
                    message="Connection failed",
                    error=error
                )
            
        except StorageProfile.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Storage profile {profile_id} not found"
            )
            
    return await test_connection_sync(profile_id, organization)


@router.post(
    "/profiles/{profile_id}/upload",
    response_model=FileUploadResponse,
    summary="Upload File",
    description="Upload a file to storage profile"
)
async def upload_file(
    profile_id: UUID,
    file: UploadFile = File(...),
    key: str = Query(..., description="Destination path/key"),
    organization=Depends(get_current_organization)
):
    """Upload a file to storage."""
    @sync_to_async
    def upload_file_sync(profile_id:UUID, organization:Organization):
        try:
            profile = StorageProfile.objects.get(storage_profile_id=profile_id, organization=organization)
            adapter = get_storage_adapter_for_profile(profile)
            
            # Upload file
            uploaded_key = adapter.upload_file(
                file.file,
                key,
                content_type=file.content_type
            )
            
            # Get file metadata
            metadata = adapter.get_file_metadata(uploaded_key)
            
            logger.info(f"Uploaded file {key} to profile {profile_id}")
            
            return FileUploadResponse(
                key=uploaded_key,
                size=metadata.size,
                content_type=metadata.content_type,
                uploaded_at=metadata.last_modified
            )
            
        except StorageProfile.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Storage profile {profile_id} not found"
            )
        except StorageError as e:
            logger.error(f"Upload failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Upload failed: {str(e)}"
            )
    return await upload_file_sync(profile_id, organization)


@router.get(
    "/profiles/{profile_id}/download/{key:path}",
    summary="Download File",
    description="Download a file from storage profile"
)
async def download_file(
    profile_id: UUID,
    key: str,
    organization=Depends(get_current_organization)
):
    """Download a file from storage."""
    @sync_to_async
    def get_file_from_storage(profile_id: UUID, key: str, organization: Organization):
        """
        Synchronous function to get file from storage.
        """
        try:
            profile = StorageProfile.objects.get(storage_profile_id=profile_id, organization=organization)
        except StorageProfile.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Storage profile {profile_id} not found"
            )
        
        adapter = get_storage_adapter_for_profile(profile)
        
        try:
            # Get file metadata first
            metadata = adapter.get_file_metadata(key)
            
            # Download file
            data = adapter.download_file(key)
            
            return {
                "data": data,
                "content_type": metadata.content_type or "application/octet-stream",
                "filename": key.split("/")[-1]
            }
            
        except StorageNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {key} not found"
            )
        except StorageError as e:
            logger.error(f"Download failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Download failed: {str(e)}"
            )
            
    result = await get_file_from_storage(profile_id, key, organization)
    
    # Return as streaming response
    from io import BytesIO
    return StreamingResponse(
        BytesIO(result["data"]),
        media_type=result["content_type"],
        headers={
            "Content-Disposition": f'attachment; filename="{result["filename"]}"'
        }
    )


@router.post(
    "/profiles/{profile_id}/list",
    response_model=FileListResponse,
    summary="List Files",
    description="List files in storage profile"
)
async def list_files(
    profile_id: UUID,
    request: FileListRequest,
    organization=Depends(get_current_organization)
):
    """List files in storage."""
    @sync_to_async
    def list_files_from_storage(profile_id: UUID, request: FileListRequest, organization: Organization):
        """
        Synchronous function to list files from storage.
        """
        try:
            profile = StorageProfile.objects.get(storage_profile_id=profile_id, organization=organization)
        except StorageProfile.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Storage profile {profile_id} not found"
            )
        
        adapter = get_storage_adapter_for_profile(profile)
        
        try:
            files = adapter.list_files(
                prefix=request.prefix,
                max_results=request.max_results
            )
            
            return {
                "files": files,
                "prefix": request.prefix
            }
            
        except StorageError as e:
            logger.error(f"List files failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"List files failed: {str(e)}"
            )

    result = await list_files_from_storage(profile_id, request, organization)
    
    from api.routers.storage.schemas import FileMetadata
    return FileListResponse(
        files=[
            FileMetadata(
                key=f.key,
                size=f.size,
                last_modified=f.last_modified,
                etag=f.etag,
                content_type=f.content_type
            )
            for f in result["files"]
        ],
        count=len(result["files"]),
        prefix=result["prefix"]
    )

@router.post(
    "/profiles/{profile_id}/presigned-url",
    response_model=PresignedUrlResponse,
    summary="Generate Presigned URL",
    description="Generate a presigned URL for temporary file access"
)
async def generate_presigned_url(
    profile_id: UUID,
    request: PresignedUrlRequest,
    organization=Depends(get_current_organization)
):
    """Generate a presigned URL."""
    @sync_to_async
    def create_presigned_url(profile_id: UUID, request: PresignedUrlRequest, organization: Organization):
        """
        Synchronous function to generate presigned URL.
        """
        try:
            profile = StorageProfile.objects.get(storage_profile_id=profile_id, organization=organization)
        except StorageProfile.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Storage profile {profile_id} not found"
            )
        
        adapter = get_storage_adapter_for_profile(profile)
        
        try:
            presigned = adapter.generate_presigned_url(
                request.key,
                expiration=request.expiration,
                method=request.method
            )
            
            return presigned
            
        except StorageError as e:
            logger.error(f"Presigned URL generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Presigned URL generation failed: {str(e)}"
            )
    
    presigned = await create_presigned_url(profile_id, request, organization)
    
    return PresignedUrlResponse(
        url=presigned.url,
        expires_at=presigned.expires_at,
        method=presigned.method
    )


@router.delete(
    "/profiles/{profile_id}/files",
    response_model=FileDeleteResponse,
    summary="Delete File",
    description="Delete a file from storage profile"
)
async def delete_file(
    profile_id: UUID,
    request: FileDeleteRequest,
    organization=Depends(get_current_organization)
):
    """Delete a file from storage."""
    @sync_to_async    
    def delete_file_from_storage(profile_id: UUID, key: str, organization: Organization):
        """
        Synchronous function to delete a file from storage.
        """
        try:
            profile = StorageProfile.objects.get(storage_profile_id=profile_id, organization=organization)
        except StorageProfile.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Storage profile {profile_id} not found"
            )
        
        adapter = get_storage_adapter_for_profile(profile)
        
        try:
            adapter.delete_file(key)
            
        except StorageNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {key} not found"
            )
        except StorageError as e:
            logger.error(f"Delete failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Delete failed: {str(e)}"
            )
    await delete_file_from_storage(profile_id, request.key, organization)
    
    logger.info(f"Deleted file {request.key} from profile {profile_id}")
    
    return FileDeleteResponse(
        success=True,
        key=request.key,
        message="File deleted successfully"
    )