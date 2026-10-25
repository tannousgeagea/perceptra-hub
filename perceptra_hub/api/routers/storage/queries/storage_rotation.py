"""
Credential rotation for storage profiles.

    PUT /api/v1/storage/profiles/{storage_profile_id}/credentials

Replaces the credential of a profile in place: the profile id never changes,
so every image registered against it keeps resolving. The new credential is
tested against the profile's config before anything is stored, the cached
adapter is dropped, and the response never echoes the secret.

Typical use: the owner of an external storage rotates its read-only key and
pushes the new one here (an admin API key works, so the rotation can be
automated by that system).
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from asgiref.sync import sync_to_async
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import RequestContext, get_request_context
from api.routers.storage.queries.storage_with_credential import StorageCredentials, prepare_credentials_dict
from perceptra_storage import StorageError, get_storage_adapter
from storage.models import EncryptedSecret, SecretRef, StorageProfile
from storage.services import create_encrypted_secret, invalidate_profile_adapter, update_encrypted_secret

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/storage", tags=["Storage Management - Combined"])


class RotateCredentialsRequest(BaseModel):
    credentials: StorageCredentials
    test_before_save: bool = Field(True, description="Verify the new credential before storing it")


class RotateCredentialsResponse(BaseModel):
    storage_profile_id: str
    name: str
    credential_type: str
    access_mode: str
    can_presign: Optional[bool] = None
    rotated_at: str
    message: str


def _rotate(org, storage_profile_id: UUID, body: RotateCredentialsRequest) -> dict:
    profile = (
        StorageProfile.objects.select_related("credential_ref")
        .filter(organization=org, storage_profile_id=storage_profile_id).first()
    )
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Storage profile not found")
    if profile.backend == "local":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Local profiles have no credentials")

    creds = body.credentials
    can_presign = None
    if creds.type == "external":
        # The secret lives in an external manager: re-point the reference;
        # rotating the value itself happens in that manager.
        new_ref = SecretRef.objects.create(
            organization=org, provider=creds.provider, path=creds.path,
            key=creds.key or "", metadata=creds.metadata or {},
        )
        old_ref = profile.credential_ref
        profile.credential_ref = new_ref
        profile.save()
        if old_ref and not old_ref.storage_profiles.exists():
            old_ref.delete()
    else:
        creds_dict = prepare_credentials_dict(creds, profile.config)
        if body.test_before_save:
            try:
                adapter = get_storage_adapter(profile.backend, profile.config, creds_dict, read_only=profile.is_external)
                adapter.test_connection()
                can_presign = adapter.can_presign
            except (StorageError, ValueError) as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"New credential failed the connection test: {e}")

        ref = profile.credential_ref
        if ref is not None and ref.provider == "local_enc" and EncryptedSecret.objects.filter(
            organization=org, identifier=ref.path,
        ).exists():
            update_encrypted_secret(org, ref.path, creds_dict)
            # profile.save() bumps updated_at: the adapter cache key changes everywhere.
            profile.save()
        else:
            identifier = f"{profile.backend}-{profile.name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
            create_encrypted_secret(org, identifier, creds_dict, description=f"Credentials for {profile.name}")
            profile.credential_ref = SecretRef.objects.create(
                organization=org, provider="local_enc", path=identifier, key="", metadata={},
            )
            profile.save()

    invalidate_profile_adapter(profile)
    logger.info("Rotated credentials of storage profile %s (%s)", profile.storage_profile_id, creds.type)
    return {"profile": profile, "credential_type": creds.type, "can_presign": can_presign}


@router.put(
    "/profiles/{storage_profile_id}/credentials",
    response_model=RotateCredentialsResponse,
    summary="Rotate storage profile credentials",
)
async def rotate_storage_credentials(
    storage_profile_id: UUID,
    body: RotateCredentialsRequest,
    ctx: RequestContext = Depends(get_request_context),
):
    ctx.require_role("admin", "owner")
    try:
        result = await sync_to_async(_rotate)(ctx.organization, storage_profile_id, body)
    except HTTPException:
        raise
    except Exception:
        # never echo the exception: it may carry the credential
        logger.exception("storage_credential_rotation_failed", extra={"profile": str(storage_profile_id)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Credential rotation failed")

    profile = result["profile"]
    return RotateCredentialsResponse(
        storage_profile_id=str(profile.storage_profile_id),
        name=profile.name,
        credential_type=result["credential_type"],
        access_mode=profile.access_mode,
        can_presign=result["can_presign"],
        rotated_at=datetime.now(timezone.utc).isoformat(),
        message="Credentials rotated",
    )
