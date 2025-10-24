"""
FastAPI routes for combined storage profile operations.

Provides simplified endpoints for creating storage profiles with credentials
in a single operation.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, Dict, Any, Literal
from uuid import UUID
from pydantic import BaseModel, Field
import logging
from asgiref.sync import sync_to_async

from api.dependencies import get_request_context, RequestContext
from storage.models import StorageProfile, SecretRef, EncryptedSecret
from organizations.models import Organization
from storage.services import (
    create_encrypted_secret,
    test_storage_profile_connection,
    get_storage_adapter_for_profile,
    StorageServiceError,
)
from perceptra_storage import (
    get_storage_adapter,
    StorageError,
    StorageConnectionError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/storage", tags=["Storage Management - Combined"])


# ============= Schemas =============

class ConnectionStringCredentials(BaseModel):
    """Credentials via connection string."""
    type: Literal["connection_string"] = "connection_string"
    connection_string: str = Field(..., description="Full connection string")


class AccountKeyCredentials(BaseModel):
    """Credentials via account key (Azure)."""
    type: Literal["account_key"] = "account_key"
    account_key: str = Field(..., description="Storage account key")


class AccessKeyCredentials(BaseModel):
    """Credentials via access/secret key (S3/MinIO)."""
    type: Literal["access_key"] = "access_key"
    access_key_id: str = Field(..., description="Access key ID")
    secret_access_key: str = Field(..., description="Secret access key")


class ExternalSecretCredentials(BaseModel):
    """Reference to external secret manager."""
    type: Literal["external"] = "external"
    provider: str = Field(..., description="Secret provider (vault, azure_kv, aws_sm)")
    path: str = Field(..., description="Path to secret")
    key: Optional[str] = Field(None, description="Specific key to extract")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ProfileConfiguration(BaseModel):
    """Storage profile configuration."""
    name: str = Field(..., min_length=1, max_length=100)
    backend: str = Field(..., description="Storage backend (azure, s3, minio, local)")
    region: Optional[str] = None
    is_default: bool = False
    config: Dict[str, Any] = Field(
        ...,
        description="Backend config (container_name, bucket_name, etc.)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Production Storage",
                "backend": "azure",
                "region": "westeurope",
                "is_default": True,
                "config": {
                    "container_name": "my-container",
                    "account_name": "mystorageaccount"
                }
            }
        }


class TestConnectionRequest(BaseModel):
    """Request to test storage connection without saving."""
    profile: ProfileConfiguration
    credentials: ConnectionStringCredentials | AccountKeyCredentials | AccessKeyCredentials | ExternalSecretCredentials
    
    class Config:
        json_schema_extra = {
            "example": {
                "profile": {
                    "name": "Test Storage",
                    "backend": "azure",
                    "config": {
                        "container_name": "test-container",
                        "account_name": "teststorage"
                    }
                },
                "credentials": {
                    "type": "connection_string",
                    "connection_string": "DefaultEndpointsProtocol=https;..."
                }
            }
        }


class TestConnectionResponse(BaseModel):
    """Response from connection test."""
    success: bool
    message: str
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class CreateProfileWithCredentialsRequest(BaseModel):
    """Request to create storage profile with credentials."""
    profile: ProfileConfiguration
    credentials: ConnectionStringCredentials | AccountKeyCredentials | AccessKeyCredentials | ExternalSecretCredentials
    test_before_save: bool = Field(
        default=True,
        description="Test connection before saving"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "profile": {
                    "name": "Production Storage",
                    "backend": "azure",
                    "config": {
                        "container_name": "prod-data",
                        "account_name": "prodaccount"
                    }
                },
                "credentials": {
                    "type": "account_key",
                    "account_key": "base64encodedkey=="
                },
                "test_before_save": True
            }
        }


class CreateProfileWithCredentialsResponse(BaseModel):
    """Response from creating storage profile with credentials."""
    storage_profile_id: str
    name: str
    backend: str
    credential_ref_id: Optional[str]
    encrypted_secret_identifier: Optional[str]
    is_default: bool
    message: str


# ============= Helper Functions =============

def prepare_credentials_dict(
    credentials: ConnectionStringCredentials | AccountKeyCredentials | AccessKeyCredentials | ExternalSecretCredentials,
    profile_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert credential schema to dictionary for storage adapter.
    
    Args:
        credentials: Credentials from request
        profile_config: Profile configuration
        
    Returns:
        Dictionary with credentials for storage adapter
    """
    if credentials.type == "connection_string":
        return {"connection_string": credentials.connection_string}
    
    elif credentials.type == "account_key":
        # For Azure - need account_name from config
        account_name = profile_config.get("account_name")
        if not account_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="account_name required in config for account_key credentials"
            )
        return {
            "account_name": account_name,
            "account_key": credentials.account_key
        }
    
    elif credentials.type == "access_key":
        # For S3/MinIO
        return {
            "access_key_id": credentials.access_key_id,
            "secret_access_key": credentials.secret_access_key
        }
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported credential type: {credentials.type}"
        )


# ============= Endpoints =============

@router.post(
    "/test-connection",
    response_model=TestConnectionResponse,
    summary="Test Storage Connection",
    description="Test storage connection without saving configuration"
)
async def test_storage_connection(
    request: TestConnectionRequest,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Test storage connection without persisting anything.
    
    This allows users to validate their configuration before saving.
    """
    try:
        # Prepare credentials
        if request.credentials.type == "external":
            # For external secrets, we can't test without creating SecretRef
            # We'll just validate the configuration structure
            return TestConnectionResponse(
                success=False,
                message="External secret validation not supported in test mode",
                error="Please save the configuration to test external secrets"
            )
        
        creds_dict = prepare_credentials_dict(
            request.credentials,
            request.profile.config
        )
        
        # Create temporary storage adapter
        adapter = get_storage_adapter(
            backend=request.profile.backend,
            config=request.profile.config,
            credentials=creds_dict
        )
        
        # Test connection
        adapter.test_connection()
        
        logger.info(
            f"Connection test successful for {request.profile.backend} "
            f"(organization: {ctx.organization.name})"
        )
        
        return TestConnectionResponse(
            success=True,
            message="Connection successful",
            details={
                "backend": request.profile.backend,
                "container": request.profile.config.get("container_name") or 
                           request.profile.config.get("bucket_name")
            }
        )
        
    except StorageConnectionError as e:
        logger.warning(f"Connection test failed: {e}")
        return TestConnectionResponse(
            success=False,
            message="Connection failed",
            error=str(e)
        )
        
    except StorageError as e:
        logger.error(f"Storage error during test: {e}")
        return TestConnectionResponse(
            success=False,
            message="Storage configuration error",
            error=str(e)
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.exception(f"Unexpected error testing connection: {e}")
        return TestConnectionResponse(
            success=False,
            message="Unexpected error",
            error=str(e)
        )


@router.post(
    "/profiles-with-credentials",
    response_model=CreateProfileWithCredentialsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Storage Profile with Credentials",
    description="Create storage profile and credentials in a single operation"
)
async def create_profile_with_credentials(
    request: CreateProfileWithCredentialsRequest,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Create storage profile with credentials in one operation.
    
    This endpoint:
    1. Creates EncryptedSecret or SecretRef based on credential type
    2. Tests connection if requested
    3. Creates StorageProfile
    4. Returns complete configuration
    
    All operations are atomic - if anything fails, nothing is saved.
    """
    # Require admin role
    ctx.require_role('admin', 'owner')
    
    @sync_to_async
    def create_complete_storage_profile(
        organization: Organization,
        profile_data: ProfileConfiguration,
        credentials: Any,
        test_first: bool
    ):
        """Synchronous function to create complete storage profile."""
        
        secret_ref = None
        encrypted_secret_id = None
        
        try:
            # Step 1: Handle credentials based on type
            if credentials.type == "external":
                # Create SecretRef for external secret manager
                secret_ref = SecretRef.objects.create(
                    organization=organization,
                    provider=credentials.provider,
                    path=credentials.path,
                    key=credentials.key or '',
                    metadata=credentials.metadata
                )
                logger.info(f"Created external SecretRef {secret_ref.id}")
                
            else:
                # Create EncryptedSecret for local storage
                creds_dict = prepare_credentials_dict(credentials, profile_data.config)
                
                # Generate unique identifier
                import uuid
                identifier = f"{profile_data.backend}-{profile_data.name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
                
                # Create encrypted secret
                encrypted_secret = create_encrypted_secret(
                    organization=organization,
                    identifier=identifier,
                    secret_data=creds_dict,
                    description=f"Credentials for {profile_data.name}"
                )
                encrypted_secret_id = identifier
                
                # Create SecretRef pointing to encrypted secret
                secret_ref = SecretRef.objects.create(
                    organization=organization,
                    provider='local_enc',
                    path=identifier,
                    key='',
                    metadata={}
                )
                logger.info(f"Created EncryptedSecret {identifier} and SecretRef {secret_ref.id}")
            
            # Step 2: Create storage profile (unsaved)
            profile = StorageProfile(
                organization=organization,
                name=profile_data.name,
                backend=profile_data.backend,
                region=profile_data.region,
                is_default=profile_data.is_default,
                config=profile_data.config,
                credential_ref=secret_ref,
                is_active=True
            )
            
            # Step 3: Test connection if requested
            if test_first:
                success, error = test_storage_profile_connection(profile)
                if not success:
                    # Cleanup created secrets
                    if secret_ref:
                        secret_ref.delete()
                    if encrypted_secret_id:
                        EncryptedSecret.objects.filter(
                            organization=organization,
                            identifier=encrypted_secret_id
                        ).delete()
                    
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Connection test failed: {error}"
                    )
            
            # Step 4: Save profile
            profile.save()
            
            logger.info(
                f"Created storage profile {profile.storage_profile_id} "
                f"for organization {organization.name}"
            )
            
            return {
                "profile": profile,
                "secret_ref": secret_ref,
                "encrypted_secret_id": encrypted_secret_id
            }
            
        except HTTPException:
            raise
            
        except Exception as e:
            # Cleanup on any error
            if secret_ref and secret_ref.id:
                try:
                    secret_ref.delete()
                except:
                    pass
            
            if encrypted_secret_id:
                try:
                    EncryptedSecret.objects.filter(
                        organization=organization,
                        identifier=encrypted_secret_id
                    ).delete()
                except:
                    pass
            
            logger.error(f"Failed to create storage profile: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create storage profile: {str(e)}"
            )
    
    try:
        result = await create_complete_storage_profile(
            ctx.organization,
            request.profile,
            request.credentials,
            request.test_before_save
        )
        
        return CreateProfileWithCredentialsResponse(
            storage_profile_id=str(result["profile"].storage_profile_id),
            name=result["profile"].name,
            backend=result["profile"].backend,
            credential_ref_id=str(result["secret_ref"].credential_ref_id),
            encrypted_secret_identifier=result["encrypted_secret_id"],
            is_default=result["profile"].is_default,
            message="Storage profile created successfully"
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.exception(f"Unexpected error creating storage profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )