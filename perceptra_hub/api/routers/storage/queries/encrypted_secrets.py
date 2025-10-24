"""
FastAPI routes for managing encrypted secrets.

Users can store credentials that will be encrypted and stored in EncryptedSecret model.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field
import logging
from cryptography.fernet import Fernet
from asgiref.sync import sync_to_async
import json

from api.dependencies import RequestContext, get_request_context, require_organization_role
from storage.models import EncryptedSecret, SecretRef
from organizations.models import Organization
from django.conf import settings
from django.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/secrets", tags=["Encrypted Secrets"])


# ============= Pydantic Schemas =============

class EncryptedSecretCreate(BaseModel):
    """Schema for creating an encrypted secret."""
    identifier: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique identifier for this secret"
    )
    secret_data: Dict[str, Any] = Field(
        ...,
        description="Secret data to encrypt (e.g., access keys, passwords)"
    )
    description: Optional[str] = Field(
        None,
        description="Description of what this secret is for"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "identifier": "s3-production-credentials",
                "secret_data": {
                    "access_key_id": "AKIAIOSFODNN7EXAMPLE",
                    "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
                },
                "description": "AWS S3 credentials for production environment"
            }
        }


class EncryptedSecretUpdate(BaseModel):
    """Schema for updating an encrypted secret."""
    secret_data: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class EncryptedSecretResponse(BaseModel):
    """Schema for encrypted secret response (WITHOUT decrypted data)."""
    id: str
    identifier: str
    description: str
    encryption_version: int
    last_decrypted_at: Optional[str]
    created_at: str
    updated_at: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "secret-uuid",
                "identifier": "s3-production-credentials",
                "description": "AWS S3 credentials for production",
                "encryption_version": 1,
                "last_decrypted_at": "2025-10-20T10:30:00Z",
                "created_at": "2025-10-20T10:00:00Z",
                "updated_at": "2025-10-20T10:00:00Z"
            }
        }


class SecretRefCreateWithEncrypted(BaseModel):
    """Schema for creating SecretRef pointing to local encrypted secret."""
    identifier: str = Field(..., description="EncryptedSecret identifier to reference")
    key: Optional[str] = Field(None, description="Specific key to extract from secret")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


# ============= Helper Functions =============

def get_encryption_key() -> bytes:
    """Get encryption key from settings."""
    encryption_key = getattr(settings, 'SECRET_ENCRYPTION_KEY', None)
    
    if not encryption_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SECRET_ENCRYPTION_KEY not configured in settings"
        )
    
    if isinstance(encryption_key, str):
        return encryption_key.encode()
    return encryption_key


def encrypt_secret_data(secret_data: Dict[str, Any]) -> str:
    """Encrypt secret data using Fernet."""
    encryption_key = get_encryption_key()
    fernet = Fernet(encryption_key)
    
    json_str = json.dumps(secret_data)
    encrypted_bytes = fernet.encrypt(json_str.encode())
    return encrypted_bytes.decode('utf-8')


# ============= CRUD Endpoints =============
@sync_to_async
def create_encrypted_secret_record(organization: Organization, secret: EncryptedSecretCreate):
    """
    Synchronous function to create an encrypted secret.
    """
    # Check if identifier already exists
    if EncryptedSecret.objects.filter(
        organization=organization,
        identifier=secret.identifier
    ).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Secret with identifier '{secret.identifier}' already exists"
        )
    
    # Encrypt the secret data
    encrypted_value = encrypt_secret_data(secret.secret_data)
    
    # Create encrypted secret
    encrypted_secret = EncryptedSecret.objects.create(
        organization=organization,
        identifier=secret.identifier,
        encrypted_value=encrypted_value,
        description=secret.description or "",
        encryption_version=1
    )
    
    return encrypted_secret


@router.post(
    "/encrypted",
    response_model=EncryptedSecretResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Encrypted Secret",
    description="Store credentials securely with encryption"
)
async def create_encrypted_secret(
    secret: EncryptedSecretCreate,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Create a new encrypted secret.
    
    The secret_data will be encrypted using Fernet and stored in the database.
    Only the identifier and metadata are stored unencrypted.
    """
    # Require admin role
    ctx.require_role('admin', 'owner')
    
    try:
        encrypted_secret = await create_encrypted_secret_record(
            ctx.organization, secret
        )
        
        logger.info(
            f"Created encrypted secret '{secret.identifier}' "
            f"for organization {ctx.organization.name}"
        )
        
        return EncryptedSecretResponse(
            id=str(encrypted_secret.id),
            identifier=encrypted_secret.identifier,
            description=encrypted_secret.description,
            encryption_version=encrypted_secret.encryption_version,
            last_decrypted_at=(
                encrypted_secret.last_decrypted_at.isoformat()
                if encrypted_secret.last_decrypted_at else None
            ),
            created_at=encrypted_secret.created_at.isoformat(),
            updated_at=encrypted_secret.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create encrypted secret: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create encrypted secret: {str(e)}"
        )


@router.get(
    "/encrypted",
    response_model=List[EncryptedSecretResponse],
    summary="List Encrypted Secrets",
    description="List all encrypted secrets for the organization"
)
async def list_encrypted_secrets(
    ctx: RequestContext = Depends(get_request_context),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """List all encrypted secrets (without decrypted values)."""
    @sync_to_async
    def get_encrypted_secrets_list(organization: Organization, skip: int, limit: int):
        """
        Synchronous function to get encrypted secrets list.
        """
        secrets = list(
            EncryptedSecret.objects.filter(
                organization=organization
            )[skip:skip + limit]
        )
        
        return secrets

    secrets = await get_encrypted_secrets_list(ctx.organization, skip, limit)
    
    return [
        EncryptedSecretResponse(
            id=str(s.id),
            identifier=s.identifier,
            description=s.description,
            encryption_version=s.encryption_version,
            last_decrypted_at=s.last_decrypted_at.isoformat() if s.last_decrypted_at else None,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat()
        )
        for s in secrets
    ]


@router.get(
    "/encrypted/{identifier}",
    response_model=EncryptedSecretResponse,
    summary="Get Encrypted Secret Info",
    description="Get encrypted secret metadata (WITHOUT decrypted value)"
)
async def get_encrypted_secret(
    identifier: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """Get encrypted secret metadata (does NOT return decrypted value)."""
    @sync_to_async
    def get_encrypted_secret_by_identifier(organization: Organization, identifier: str):
        """
        Synchronous function to get encrypted secret by identifier.
        """
        try:
            secret = EncryptedSecret.objects.get(
                organization=organization,
                identifier=identifier
            )
            return secret
            
        except EncryptedSecret.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Encrypted secret '{identifier}' not found"
            )
    
    secret = await get_encrypted_secret_by_identifier(
        ctx.organization, identifier
    )
        
    return EncryptedSecretResponse(
        id=str(secret.id),
        identifier=secret.identifier,
        description=secret.description,
        encryption_version=secret.encryption_version,
        last_decrypted_at=secret.last_decrypted_at.isoformat() if secret.last_decrypted_at else None,
        created_at=secret.created_at.isoformat(),
        updated_at=secret.updated_at.isoformat()
    )
        


@router.put(
    "/encrypted/{identifier}",
    response_model=EncryptedSecretResponse,
    summary="Update Encrypted Secret",
    description="Update encrypted secret data or description"
)
async def update_encrypted_secret(
    identifier: str,
    update_data: EncryptedSecretUpdate,
    ctx: RequestContext = Depends(get_request_context)
):
    """Update an encrypted secret."""
    
    # Require admin role
    ctx.require_role('admin', 'owner')

    @sync_to_async
    def update_encrypted_secret_record(organization: Organization, identifier: str, update_data: EncryptedSecretUpdate):
        """
        Synchronous function to update encrypted secret.
        """
        try:
            secret = EncryptedSecret.objects.get(
                organization=organization,
                identifier=identifier
            )
        except EncryptedSecret.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Encrypted secret '{identifier}' not found"
            )
        
        # Track if secret data was updated (for logging)
        secret_data_updated = False
        
        # Update secret data if provided
        if update_data.secret_data is not None:
            encrypted_value = encrypt_secret_data(update_data.secret_data)
            secret.encrypted_value = encrypted_value
            secret_data_updated = True
        
        # Update description if provided
        if update_data.description is not None:
            secret.description = update_data.description
        
        secret.save()
        
        return secret, secret_data_updated

    secret, secret_data_updated = await update_encrypted_secret_record(
        ctx.organization, identifier, update_data
    )
    
    if secret_data_updated:
        logger.info(f"Updated encrypted value for secret '{identifier}'")
    
    return EncryptedSecretResponse(
        id=str(secret.id),
        identifier=secret.identifier,
        description=secret.description,
        encryption_version=secret.encryption_version,
        last_decrypted_at=secret.last_decrypted_at.isoformat() if secret.last_decrypted_at else None,
        created_at=secret.created_at.isoformat(),
        updated_at=secret.updated_at.isoformat()
    )


@router.delete(
    "/encrypted/{identifier}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Encrypted Secret",
    description="Delete an encrypted secret"
)
async def delete_encrypted_secret(
    identifier: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """Delete an encrypted secret."""
    
    # Require admin role
    ctx.require_role('admin', 'owner')
    
    @sync_to_async
    def delete_encrypted_secret_record(organization:Organization, identifier:str):    
        try:
            secret = EncryptedSecret.objects.get(
                organization=organization,
                identifier=identifier
            )
            
            # Check if any SecretRefs are using this
            secret_refs_using = SecretRef.objects.filter(
                organization=ctx.organization,
                provider='local_enc',
                path=identifier
            )
            
            if secret_refs_using.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Cannot delete secret '{identifier}' - "
                        f"it is referenced by {secret_refs_using.count()} SecretRef(s)"
                    )
                )
            
            secret.delete()
            logger.info(f"Deleted encrypted secret '{identifier}'")
            
        except EncryptedSecret.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Encrypted secret '{identifier}' not found"
            )
        
    await delete_encrypted_secret_record(organization=ctx.organization, identifier=identifier)


# ============= Create SecretRef pointing to Encrypted Secret =============

@router.post(
    "/encrypted/{identifier}/create-ref",
    status_code=status.HTTP_201_CREATED,
    summary="Create SecretRef for Encrypted Secret",
    description="Create a SecretRef that points to this encrypted secret"
)
async def create_secret_ref_for_encrypted(
    identifier: str,
    ref_data: SecretRefCreateWithEncrypted,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Create a SecretRef that points to an encrypted secret.
    
    This allows you to use the encrypted secret with storage profiles.
    """

    @sync_to_async
    def create_secret_ref_for_encrypted_secret(organization: Organization, identifier: str, ref_data: SecretRefCreateWithEncrypted):
        """
        Synchronous function to create a SecretRef for an encrypted secret.
        """
        # Verify encrypted secret exists
        try:
            encrypted_secret = EncryptedSecret.objects.get(
                organization=organization,
                identifier=identifier
            )
        except EncryptedSecret.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Encrypted secret '{identifier}' not found"
            )
        
        # Create SecretRef
        secret_ref = SecretRef.objects.create(
            organization=organization,
            provider='local_enc',
            path=identifier,  # The path is the identifier of the EncryptedSecret
            key=ref_data.key or '',
            metadata=ref_data.metadata
        )
        
        return secret_ref
    
    secret_ref = await create_secret_ref_for_encrypted_secret(
        ctx.organization, identifier, ref_data
    )
    
    logger.info(
        f"Created SecretRef {secret_ref.id} pointing to "
        f"encrypted secret '{identifier}'"
    )
    
    return {
        "message": "SecretRef created successfully",
        "secret_ref_id": str(secret_ref.id),
        "encrypted_secret_identifier": identifier,
        "provider": "local_enc",
        "path": identifier
    }


# ============= Test/Verify Secret =============

@router.post(
    "/encrypted/{identifier}/test",
    summary="Test Encrypted Secret",
    description="Verify that secret can be decrypted (does NOT return decrypted value)"
)
async def test_encrypted_secret(
    identifier: str,
    ctx: RequestContext = Depends(get_request_context)
):
    """
    Test if encrypted secret can be successfully decrypted.
    
    Returns success/failure but does NOT return the decrypted value.
    """
    @sync_to_async
    def test_decrypt_encrypted_secret(organization: Organization, identifier: str):
        """
        Synchronous function to test encrypted secret decryption.
        """
        try:
            secret = EncryptedSecret.objects.get(
                organization=organization,
                identifier=identifier
            )
        except EncryptedSecret.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Encrypted secret '{identifier}' not found"
            )
        
        try:
            # Try to decrypt
            encryption_key = get_encryption_key()
            decrypted_value = secret.get_decrypted_value(encryption_key)
            
            # Update last decrypted timestamp
            secret.update_last_decrypted(commit=True)
            
            # Return info WITHOUT the actual value
            return {
                "success": True,
                "identifier": identifier,
                "has_keys": list(decrypted_value.keys()) if isinstance(decrypted_value, dict) else None
            }
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Decryption failed: {str(e)}"
            )
    
    result = await test_decrypt_encrypted_secret(
        ctx.organization, identifier
    )
    
    return {
        "success": result["success"],
        "message": "Secret decrypted successfully",
        "identifier": result["identifier"],
        "has_keys": result["has_keys"]
    }