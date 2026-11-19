"""
Managing an organization's image tags and model tags.

    GET    /api/v1/tags/{kind}              every tag with how often it is used
                                            (?project_id=… → only tags used in that project)
    PATCH  /api/v1/tags/{kind}/{tag_id}     rename, recolour, describe; renaming
                                            into an existing name needs merge=true
    DELETE /api/v1/tags/{kind}/{tag_id}     delete the tag and take it off everything

kind is "image" or "model". Tags belong to the organization, so a change made
from a project's view applies everywhere. Reading: any member (a project's
view: the project's view permission). Changing: org editor and up.

Tags the platform sets itself keep their name: "archived" (similarity archive)
and "baseline" (model versions). They can be recoloured and described, not
renamed, merged or deleted, and no other tag can take their name.
"""
import re
from typing import List, Literal, Optional
from uuid import UUID

from django.db import transaction
from django.db.models import Count, Q
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.dependencies import RequestContext, get_request_context, require_permission
from images.models import ImageTag, Tag
from ml_models.models import ModelTag
from projects.dependencies import require_project_permission
from projects.models import Project

router = APIRouter()

TagKind = Literal["image", "model"]

PROTECTED = {"image": {"archived"}, "model": {"baseline"}}
MAX_NAME = {"image": Tag._meta.get_field("name").max_length,
            "model": ModelTag._meta.get_field("name").max_length}
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class TagUsage(BaseModel):
    images: int = 0
    models: int = 0
    versions: int = 0


class ManagedTag(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    color: str
    protected: bool
    usage: TagUsage


class TagUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = Field(None, max_length=2000)
    # Renaming into a name another tag already has merges this tag into it.
    merge: bool = False


def _model_for(kind: str):
    return Tag if kind == "image" else ModelTag


def _project(ctx: RequestContext, project_id: UUID) -> Project:
    project = Project.objects.filter(
        project_id=project_id, organization=ctx.organization, is_deleted=False
    ).first()
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    require_project_permission(ctx, project, "view", "Viewing this project's tags")
    return project


def _with_usage(kind: str, qs, project: Optional[Project]):
    if kind == "image":
        in_scope = Q(tagged_images__image__project_assignments__project=project,
                     tagged_images__image__project_assignments__is_active=True) if project else Q()
        return qs.annotate(n_images=Count("tagged_images__image", filter=in_scope, distinct=True))
    models_scope = Q(models__project=project) if project else Q()
    versions_scope = Q(model_versions__model__project=project) if project else Q()
    return qs.annotate(
        n_models=Count("models", filter=models_scope & Q(models__is_deleted=False), distinct=True),
        n_versions=Count("model_versions", filter=versions_scope & Q(model_versions__is_deleted=False),
                         distinct=True),
    )


def _serialize(kind: str, tag) -> ManagedTag:
    usage = (TagUsage(images=tag.n_images) if kind == "image"
             else TagUsage(models=tag.n_models, versions=tag.n_versions))
    return ManagedTag(
        id=tag.id, name=tag.name, description=tag.description or "", color=tag.color,
        protected=tag.name in PROTECTED[kind], usage=usage,
    )


def _get_tag(kind: str, ctx: RequestContext, tag_id: int, lock: bool = False):
    qs = _model_for(kind).objects.filter(organization=ctx.organization, id=tag_id)
    tag = (qs.select_for_update() if lock else qs).first()
    if tag is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tag not found")
    return tag


def _usage_of(kind: str, tag) -> ManagedTag:
    return _serialize(kind, _with_usage(kind, type(tag).objects.filter(id=tag.id), None).get())


def _merge(kind: str, source, target) -> None:
    """Everything tagged `source` gets `target`; `source` is deleted."""
    if kind == "image":
        already = ImageTag.objects.filter(tag=target).values("image")
        ImageTag.objects.filter(tag=source, image__in=already).delete()
        ImageTag.objects.filter(tag=source).update(tag=target)
    else:
        target.models.add(*source.models.all())
        target.model_versions.add(*source.model_versions.all())
    source.delete()


@router.get("/tags/{kind}", response_model=List[ManagedTag])
def list_managed_tags(
    kind: TagKind,
    project_id: Optional[UUID] = Query(None, description="Only tags used in this project"),
    ctx: RequestContext = Depends(get_request_context),
):
    """Every tag of this kind in the organization, or those used in one project."""
    project = _project(ctx, project_id) if project_id else None
    qs = _with_usage(kind, _model_for(kind).objects.filter(organization=ctx.organization), project)
    if project is not None:
        qs = qs.filter(n_images__gt=0) if kind == "image" else qs.filter(Q(n_models__gt=0) | Q(n_versions__gt=0))
    return [_serialize(kind, t) for t in qs.order_by("name")]


@router.patch("/tags/{kind}/{tag_id}", response_model=ManagedTag)
def update_managed_tag(
    kind: TagKind,
    tag_id: int,
    data: TagUpdate,
    ctx: RequestContext = Depends(require_permission("write")),
):
    """
    Rename, recolour or describe a tag (org editor and up). Renaming into a name
    another tag has returns 409 with that tag, unless merge=true: then this tag
    is merged into it and the merged tag is returned.
    """
    protected = PROTECTED[kind]
    with transaction.atomic():
        tag = _get_tag(kind, ctx, tag_id, lock=True)

        if data.color is not None:
            if not COLOR_RE.match(data.color):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail="Colour must be a hex value like #3B82F6")
            tag.color = data.color
        if data.description is not None:
            tag.description = data.description.strip()

        name = data.name.strip() if data.name is not None else tag.name
        if name != tag.name:
            if not name:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tag name can't be empty")
            if len(name) > MAX_NAME[kind]:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"Tag names are at most {MAX_NAME[kind]} characters")
            if tag.name in protected:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f'"{tag.name}" is set by the platform and can\'t be renamed')
            if name.lower() in protected:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f'"{name}" is reserved for the platform')

            # Names are unique per organization; differing only in case counts as the same tag.
            other = (_model_for(kind).objects
                     .filter(organization=ctx.organization, name__iexact=name)
                     .exclude(id=tag.id).first())
            if other is not None:
                if not data.merge:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "message": f'A tag named "{other.name}" already exists',
                            "existing": _usage_of(kind, other).model_dump(),
                        },
                    )
                _merge(kind, tag, other)
                return _usage_of(kind, other)
            tag.name = name

        tag.save()
        return _usage_of(kind, tag)


@router.delete("/tags/{kind}/{tag_id}")
def delete_managed_tag(
    kind: TagKind,
    tag_id: int,
    ctx: RequestContext = Depends(require_permission("write")),
):
    """Delete a tag and remove it from every image or model (org editor and up)."""
    with transaction.atomic():
        tag = _get_tag(kind, ctx, tag_id, lock=True)
        if tag.name in PROTECTED[kind]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f'"{tag.name}" is set by the platform and can\'t be deleted')
        usage = _usage_of(kind, tag).usage
        tag.delete()
    return {"deleted": tag_id, "removed_from": usage.model_dump()}
