"""
Rollouts: moving devices to a model version.

Starting, pausing, resuming or rolling back needs "manage" in the model's
project, the same rule as promoting a version. Seeing them needs "view".
"""
from typing import List, Literal, Optional
from uuid import UUID

from django.db.models import Prefetch
from django.utils import timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.dependencies import RequestContext, get_request_context

router = APIRouter()


class RolloutIn(BaseModel):
    model_id: str = Field(..., max_length=255)
    version_id: str = Field(..., max_length=255)
    device_ids: List[UUID] = Field(default_factory=list, max_length=5000)
    site_ids: List[UUID] = Field(default_factory=list, max_length=500)
    labels: List[str] = Field(default_factory=list, max_length=50)
    running_model: bool = Field(False, description="Every device that runs (or should run) this model now")
    strategy: Literal["all", "canary"] = "all"
    canary_count: int = Field(1, ge=1, le=1000)
    canary_minutes: int = Field(15, ge=0, le=1440)
    max_failures: int = Field(0, ge=0, le=10000)


def _require(ctx, project, perm, what):
    from projects.dependencies import require_project_permission
    require_project_permission(ctx, project, perm, what)


def _model(ctx, model_id):
    from ml_models.models import Model
    m = Model.objects.select_related("project").filter(model_id=model_id, organization=ctx.organization, is_deleted=False).first()
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return m


def _rollout(ctx, rollout_id):
    from fleet.models import Rollout
    r = (Rollout.objects.select_related("model__project", "version", "previous_version", "created_by", "organization")
         .filter(rollout_id=rollout_id, organization=ctx.organization).first())
    if r is None:
        raise HTTPException(status_code=404, detail="Rollout not found")
    return r


def _fleet_error(e):
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get("/fleet/rollouts")
def list_rollouts(
    model_id: Optional[str] = Query(None),
    project_id: Optional[UUID] = Query(None),
    active: bool = Query(False, description="Only rollouts still in progress or paused"),
    limit: int = Query(20, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context),
):
    from fleet import serializers, services
    from fleet.models import Rollout
    from memberships.roles import ORG_ROLES_WITH_PROJECT_BYPASS

    qs = Rollout.objects.filter(organization=ctx.organization).select_related(
        "model__project", "version", "previous_version", "created_by")
    if not (ctx.role and ctx.role.name in ORG_ROLES_WITH_PROJECT_BYPASS):
        # Only projects the person belongs to (every project role can view).
        qs = qs.filter(model__project__memberships__user=ctx.user) if ctx.user else qs.none()
    if model_id:
        qs = qs.filter(model__model_id=model_id)
    if project_id:
        qs = qs.filter(model__project__project_id=project_id)
    if active:
        qs = qs.filter(status__in=services.ACTIVE)
    now = timezone.now()
    return [serializers.rollout(r, now) for r in qs.distinct()[:limit]]


@router.post("/fleet/rollouts", status_code=201)
def create_rollout(body: RolloutIn, ctx: RequestContext = Depends(get_request_context)):
    from fleet import serializers, services
    from ml_models.models import ModelVersion

    model = _model(ctx, body.model_id)
    _require(ctx, model.project, "manage", "Deploying a model to devices")
    version = ModelVersion.objects.filter(model=model, version_id=body.version_id, is_deleted=False).first()
    if version is None:
        raise HTTPException(status_code=404, detail="Version not found")
    devices = services.select_devices(ctx.organization, model=model, device_ids=body.device_ids, site_ids=body.site_ids,
                                      labels=body.labels, running_model=body.running_model)
    selector = {"device_ids": [str(d) for d in body.device_ids], "site_ids": [str(s) for s in body.site_ids],
                "labels": body.labels, "running_model": body.running_model}
    try:
        r = services.create_rollout(ctx.organization, model, version, devices, user=ctx.user, strategy=body.strategy,
                                    canary_count=body.canary_count, canary_minutes=body.canary_minutes,
                                    max_failures=body.max_failures, selector=selector)
    except services.FleetError as e:
        raise _fleet_error(e)
    return serializers.rollout(_rollout(ctx, r.rollout_id), targets=True)


@router.get("/fleet/rollouts/{rollout_id}")
def get_rollout(rollout_id: UUID, ctx: RequestContext = Depends(get_request_context)):
    from fleet import serializers
    r = _rollout(ctx, rollout_id)
    _require(ctx, r.model.project, "view", "Viewing this rollout")
    return serializers.rollout(r, targets=True)


def _act(ctx, rollout_id, action):
    from fleet import serializers, services
    r = _rollout(ctx, rollout_id)
    _require(ctx, r.model.project, "manage", "Changing a rollout")
    try:
        result = action(services, r)
    except services.FleetError as e:
        raise _fleet_error(e)
    return serializers.rollout(_rollout(ctx, result.rollout_id), targets=True)


@router.post("/fleet/rollouts/{rollout_id}/pause")
def pause_rollout(rollout_id: UUID, ctx: RequestContext = Depends(get_request_context)):
    name = (ctx.user.get_full_name() or ctx.user.email) if ctx.user else "an API key"
    return _act(ctx, rollout_id, lambda s, r: s.pause(r, f"Paused by {name}."))


@router.post("/fleet/rollouts/{rollout_id}/resume")
def resume_rollout(rollout_id: UUID, ctx: RequestContext = Depends(get_request_context)):
    return _act(ctx, rollout_id, lambda s, r: s.resume(r))


@router.post("/fleet/rollouts/{rollout_id}/rollback", status_code=201)
def rollback_rollout(rollout_id: UUID, ctx: RequestContext = Depends(get_request_context)):
    """Returns the new rollout that puts the devices back on the previous version."""
    return _act(ctx, rollout_id, lambda s, r: s.rollback(r, user=ctx.user))


@router.get("/fleet/models/{model_id}/devices")
def model_devices(model_id: str, ctx: RequestContext = Depends(get_request_context)):
    """Every device that runs or should run this model, for the project's Deploy view."""
    from fleet import serializers
    from fleet.models import Device, DeviceModel

    model = _model(ctx, model_id)
    _require(ctx, model.project, "view", "Viewing this model's devices")
    devices = (Device.objects.filter(organization=ctx.organization, models__model=model).distinct().select_related("site")
               .prefetch_related("cameras", Prefetch("models", queryset=DeviceModel.objects.filter(model=model).select_related(
                   "model__project", "desired_version", "reported_version", "rollout"))))
    now = timezone.now()
    return [serializers.device(d, now) for d in devices]
