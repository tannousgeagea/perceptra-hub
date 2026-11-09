"""
Tags on one model version.

    PUT    /api/v1/models/{model_id}/versions/{version_id}/tags/{tag}   add
    DELETE /api/v1/models/{model_id}/versions/{version_id}/tags/{tag}   remove

Only "baseline" for now: one version per model (setting it moves it). Where a
version runs (staging / production) is its deployment status, set by Deploy,
not a tag.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from asgiref.sync import sync_to_async
from django.db import transaction

from api.dependencies import RequestContext, get_request_context
from api.routers.ml_models.schemas import ModelVersionResponse
from api.routers.ml_models.utils import get_model_by_id, serialize_model_version
from ml_models.models import Model, ModelTag, ModelVersion

router = APIRouter(prefix="/models")

ALLOWED_TAGS = {"baseline"}
# At most one version of a model carries these.
EXCLUSIVE_TAGS = {"baseline"}


def _check_tag(tag: str) -> str:
    tag = tag.strip().lower()
    if tag not in ALLOWED_TAGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown version tag '{tag}'. Allowed: {', '.join(sorted(ALLOWED_TAGS))}",
        )
    return tag


def _check_can_edit(model: Model, ctx: RequestContext) -> None:
    # Same rule as editing the model: its creator, or an org admin/owner.
    if not ctx.has_role('admin', 'owner') and model.created_by_id != ctx.user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the model's creator or an organization admin can tag its versions",
        )


def _tag_object(organization, name: str) -> ModelTag:
    # ModelTag.name is unique across organizations, so an existing tag of that
    # name is reused wherever it was first created.
    tag = ModelTag.objects.filter(name=name).first()
    return tag or ModelTag.objects.create(organization=organization, name=name)


@sync_to_async
def _set_tag(model: Model, version_id: str, tag: str, value: bool) -> dict:
    with transaction.atomic():
        version = (
            ModelVersion.objects.select_for_update()
            .filter(model=model, version_id=version_id, is_deleted=False)
            .first()
        )
        if version is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Version {version_id} not found")
        tag_obj = _tag_object(model.organization, tag)
        if value:
            if tag in EXCLUSIVE_TAGS:
                for other in model.versions.filter(tags=tag_obj).exclude(pk=version.pk):
                    other.tags.remove(tag_obj)
            version.tags.add(tag_obj)
        else:
            version.tags.remove(tag_obj)
    version = ModelVersion.objects.select_related(
        'dataset_version__project', 'created_by', 'storage_profile'
    ).get(pk=version.pk)
    return serialize_model_version(version)


@router.put("/{model_id}/versions/{version_id}/tags/{tag}", response_model=ModelVersionResponse)
async def add_version_tag(
    model_id: str,
    version_id: str,
    tag: str,
    ctx: RequestContext = Depends(get_request_context),
):
    """Tag a model version (for "baseline", untagging the model's other versions)."""
    tag = _check_tag(tag)
    model = await get_model_by_id(model_id, ctx.organization)
    _check_can_edit(model, ctx)
    return await _set_tag(model, version_id, tag, True)


@router.delete("/{model_id}/versions/{version_id}/tags/{tag}", response_model=ModelVersionResponse)
async def remove_version_tag(
    model_id: str,
    version_id: str,
    tag: str,
    ctx: RequestContext = Depends(get_request_context),
):
    """Remove a tag from a model version."""
    tag = _check_tag(tag)
    model = await get_model_by_id(model_id, ctx.organization)
    _check_can_edit(model, ctx)
    return await _set_tag(model, version_id, tag, False)
