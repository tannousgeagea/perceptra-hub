"""
Tags on one model version.

    PUT    /api/v1/models/{model_id}/versions/{version_id}/tags/{tag}   add
    DELETE /api/v1/models/{model_id}/versions/{version_id}/tags/{tag}   remove

Any of the organization's model tags; a new name creates one (stored lower-case,
like tags typed on a model). "baseline" is special: one version per model
(setting it moves it). "production" and "staging" are refused: where a version
runs is its deployment status, set by Deploy, so a tag could only contradict it.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from asgiref.sync import sync_to_async
from django.db import transaction

from api.dependencies import RequestContext, get_request_context
from api.routers.ml_models.schemas import ModelVersionResponse
from api.routers.ml_models.utils import get_model_by_id, serialize_model_version
from ml_models.models import Model, ModelTag, ModelVersion
from projects.dependencies import require_project_permission

router = APIRouter(prefix="/models")

# At most one version of a model carries these.
EXCLUSIVE_TAGS = {"baseline"}
# Deployment states: shown from deployment_status, never stored as tags.
DEPLOYMENT_NAMES = {"production", "staging"}
MAX_TAG_LENGTH = ModelTag._meta.get_field("name").max_length


def _check_tag(tag: str, adding: bool = True) -> str:
    tag = tag.strip().lower()
    if not adding:
        # Removing: any name, so tags from before these rules can still be taken off.
        if not tag:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tag name can't be empty")
        return tag
    if not tag:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tag name can't be empty")
    if len(tag) > MAX_TAG_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tag names are at most {MAX_TAG_LENGTH} characters",
        )
    if tag in DEPLOYMENT_NAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'"{tag}" is where a version runs: set it with Deploy, not a tag',
        )
    return tag


async def _check_can_edit(model: Model, ctx: RequestContext) -> None:
    # Same rule as editing the model: the editor role in the model's project.
    await sync_to_async(require_project_permission)(ctx, model.project, "edit", "Tagging a model version")


def _find_tag(organization, name: str):
    # Tag names are unique per organization; differing only in case is the same tag.
    return ModelTag.objects.filter(organization=organization, name__iexact=name).order_by("id").first()


def _tag_object(organization, name: str) -> ModelTag:
    tag = _find_tag(organization, name)
    if tag is None:
        tag, _ = ModelTag.objects.get_or_create(organization=organization, name=name)
    return tag


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
        if value:
            tag_obj = _tag_object(model.organization, tag)
            if tag in EXCLUSIVE_TAGS:
                for other in model.versions.filter(tags=tag_obj).exclude(pk=version.pk):
                    other.tags.remove(tag_obj)
            version.tags.add(tag_obj)
        else:
            # Removing never creates a tag; a name the org doesn't have is a no-op.
            tag_obj = _find_tag(model.organization, tag)
            if tag_obj is not None:
                version.tags.remove(tag_obj)
    version = ModelVersion.objects.select_related(
        'dataset_version__project', 'created_by', 'storage_profile'
    ).get(pk=version.pk)
    return serialize_model_version(version)


# :path so a tag name may contain "/".
@router.put("/{model_id}/versions/{version_id}/tags/{tag:path}", response_model=ModelVersionResponse)
async def add_version_tag(
    model_id: str,
    version_id: str,
    tag: str,
    ctx: RequestContext = Depends(get_request_context),
):
    """Tag a model version (for "baseline", untagging the model's other versions)."""
    tag = _check_tag(tag)
    model = await get_model_by_id(model_id, ctx.organization)
    await _check_can_edit(model, ctx)
    return await _set_tag(model, version_id, tag, True)


@router.delete("/{model_id}/versions/{version_id}/tags/{tag:path}", response_model=ModelVersionResponse)
async def remove_version_tag(
    model_id: str,
    version_id: str,
    tag: str,
    ctx: RequestContext = Depends(get_request_context),
):
    """Remove a tag from a model version."""
    tag = _check_tag(tag, adding=False)
    model = await get_model_by_id(model_id, ctx.organization)
    await _check_can_edit(model, ctx)
    return await _set_tag(model, version_id, tag, False)
