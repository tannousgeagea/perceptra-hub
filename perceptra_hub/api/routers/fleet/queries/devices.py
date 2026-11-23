"""
Sites, devices and cameras, for the console.

Seeing the fleet: any organization member ("read"). Adding, changing or
revoking devices and sites: organization admins ("admin"), because a device
key gives a machine access to the organization's models.
"""
from datetime import timedelta
from typing import List, Optional
from uuid import UUID

from django.db import IntegrityError
from django.db.models import Prefetch
from django.utils import timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from organizations.dependencies import require_permission

router = APIRouter()

READ = Depends(require_permission("read"))
ADMIN = Depends(require_permission("admin"))


class SiteIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    location: str = Field("", max_length=255)
    timezone: str = Field("", max_length=64)


class SitePatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    location: Optional[str] = Field(None, max_length=255)
    timezone: Optional[str] = Field(None, max_length=64)


class DeviceIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    site_id: Optional[UUID] = None
    labels: List[str] = Field(default_factory=list, max_length=20)


class DevicePatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    site_id: Optional[UUID] = None
    clear_site: bool = False
    labels: Optional[List[str]] = Field(None, max_length=20)
    poll_seconds: Optional[int] = Field(None, ge=10, le=3600)


class CameraIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    position: str = Field("", max_length=120)
    order: int = Field(0, ge=0)
    settings: dict = Field(default_factory=dict)


class CameraPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    position: Optional[str] = Field(None, max_length=120)
    order: Optional[int] = Field(None, ge=0)
    settings: Optional[dict] = None


def _clean_labels(labels):
    seen = []
    for label in labels or []:
        label = str(label).strip()[:40]
        if label and label not in seen:
            seen.append(label)
    return seen


def _site(ctx, site_id):
    from fleet.models import Site
    s = Site.objects.filter(organization=ctx.organization, site_id=site_id).first()
    if s is None:
        raise HTTPException(status_code=404, detail="Site not found")
    return s


def _devices_qs(ctx):
    from fleet.models import Device, DeviceModel
    return (Device.objects.filter(organization=ctx.organization).select_related("site")
            .prefetch_related("cameras", Prefetch("models", queryset=DeviceModel.objects.select_related(
                "model__project", "desired_version", "reported_version", "rollout").order_by("model__name"))))


def _device(ctx, device_id):
    d = _devices_qs(ctx).filter(device_id=device_id).first()
    if d is None:
        raise HTTPException(status_code=404, detail="Device not found")
    return d


def _conflict(what):
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"{what} with this name already exists in the organization.")


def _install(code: str) -> dict:
    """The one-time code, shown once. The device runs: malumetrix-edge enroll --url <server> --code <code>."""
    from fleet.keys import CODE_TTL
    return {"code": code, "expires_at": (timezone.now() + CODE_TTL).isoformat()}


# ── Sites ────────────────────────────────────────────────────────────────────

@router.get("/fleet/sites")
def list_sites(ctx=READ):
    from fleet import serializers, services
    from fleet.models import Device, Site

    now = timezone.now()
    counts = {}
    for d in Device.objects.filter(organization=ctx.organization, site__isnull=False).only(
            "site_id", "revoked_at", "enrolled_at", "last_seen_at", "poll_seconds"):
        c = counts.setdefault(d.site_id, {"total": 0, "online": 0, "stale": 0, "offline": 0, "pending": 0, "revoked": 0})
        c["total"] += 1
        c[services.device_status(d, now)] += 1
    return [serializers.site(s, counts.get(s.id, {"total": 0, "online": 0, "stale": 0, "offline": 0, "pending": 0, "revoked": 0}))
            for s in Site.objects.filter(organization=ctx.organization)]


@router.post("/fleet/sites", status_code=201)
def create_site(body: SiteIn, ctx=ADMIN):
    from fleet import serializers
    from fleet.models import Site
    try:
        s = Site.objects.create(organization=ctx.organization, name=body.name.strip(), location=body.location.strip(),
                                timezone=body.timezone.strip(), created_by=ctx.user)
    except IntegrityError:
        raise _conflict("A site")
    return serializers.site(s)


@router.patch("/fleet/sites/{site_id}")
def update_site(site_id: UUID, body: SitePatch, ctx=ADMIN):
    from fleet import serializers
    s = _site(ctx, site_id)
    for field in ("name", "location", "timezone"):
        value = getattr(body, field)
        if value is not None:
            setattr(s, field, value.strip())
    try:
        s.save()
    except IntegrityError:
        raise _conflict("A site")
    return serializers.site(s)


@router.delete("/fleet/sites/{site_id}", status_code=204)
def delete_site(site_id: UUID, ctx=ADMIN):
    """Its devices stay, without a site."""
    _site(ctx, site_id).delete()


# ── Devices ──────────────────────────────────────────────────────────────────

@router.get("/fleet/devices")
def list_devices(
    site_id: Optional[UUID] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status", description="online, stale, offline, pending, revoked"),
    model_id: Optional[str] = Query(None, description="Devices that run or should run this model"),
    label: Optional[str] = Query(None),
    q: Optional[str] = Query(None, max_length=120),
    ctx=READ,
):
    from fleet import serializers, services

    qs = _devices_qs(ctx)
    if site_id:
        qs = qs.filter(site__site_id=site_id)
    if model_id:
        qs = qs.filter(models__model__model_id=model_id).distinct()
    if label:
        qs = qs.filter(labels__contains=[label])
    if q:
        qs = qs.filter(name__icontains=q)
    now = timezone.now()
    devices = [d for d in qs if not status_filter or services.device_status(d, now) == status_filter]
    return [serializers.device(d, now) for d in devices]


@router.post("/fleet/devices", status_code=201)
def create_device(body: DeviceIn, ctx=ADMIN):
    """A pending device and its one-time enrollment code (shown only in this answer)."""
    from fleet import serializers, services
    from fleet.models import Device

    site = _site(ctx, body.site_id) if body.site_id else None
    try:
        d = Device.objects.create(organization=ctx.organization, site=site, name=body.name.strip(),
                                  labels=_clean_labels(body.labels), created_by=ctx.user)
    except IntegrityError:
        raise _conflict("A device")
    code = services.issue_enrollment_code(d)
    return {"device": serializers.device(_device(ctx, d.device_id)), "enrollment": _install(code)}


@router.get("/fleet/devices/{device_id}")
def get_device(device_id: UUID, ctx=READ):
    from fleet import serializers
    from fleet.models import RolloutTarget

    d = _device(ctx, device_id)
    out = serializers.device(d, detail=True)
    out["rollouts"] = [{
        "rollout_id": str(t.rollout.rollout_id), "model_name": t.rollout.model.name,
        "project_id": str(t.rollout.model.project.project_id),
        # The version this device was asked to run (a rollback sends devices to different ones).
        "version": serializers.version_ref(t.to_version or t.rollout.version), "is_rollback": t.rollout.is_rollback,
        "from_version": serializers.version_ref(t.from_version),
        "rollout_status": t.rollout.status, "state": t.state, "error": t.error or None,
        "created_at": serializers.iso(t.rollout.created_at), "finished_at": serializers.iso(t.finished_at),
    } for t in RolloutTarget.objects.filter(device=d).select_related("rollout__model__project", "rollout__version", "to_version", "from_version")
        .order_by("-rollout__created_at")[:20]]
    return out


@router.patch("/fleet/devices/{device_id}")
def update_device(device_id: UUID, body: DevicePatch, ctx=ADMIN):
    from fleet import serializers
    d = _device(ctx, device_id)
    if body.name is not None:
        d.name = body.name.strip()
    if body.clear_site:
        d.site = None
    elif body.site_id:
        d.site = _site(ctx, body.site_id)
    if body.labels is not None:
        d.labels = _clean_labels(body.labels)
    if body.poll_seconds is not None:
        d.poll_seconds = body.poll_seconds
    try:
        d.save()
    except IntegrityError:
        raise _conflict("A device")
    return serializers.device(_device(ctx, device_id), detail=True)


@router.post("/fleet/devices/{device_id}/enrollment-code")
def new_enrollment_code(device_id: UUID, ctx=ADMIN):
    """A fresh code: to set up a replacement machine, or after revoking. The current key keeps working until the code is used."""
    from fleet import services
    return {"enrollment": _install(services.issue_enrollment_code(_device(ctx, device_id)))}


@router.post("/fleet/devices/{device_id}/revoke")
def revoke_device(device_id: UUID, ctx=ADMIN):
    from fleet import serializers, services
    d = _device(ctx, device_id)
    if d.revoked_at:
        raise HTTPException(status_code=409, detail="This device is already revoked.")
    services.revoke(d)
    return serializers.device(_device(ctx, device_id), detail=True)


@router.delete("/fleet/devices/{device_id}", status_code=204)
def delete_device(device_id: UUID, ctx=ADMIN):
    """Only devices that can't call any more (revoked, or never enrolled)."""
    d = _device(ctx, device_id)
    if d.enrolled_at and not d.revoked_at:
        raise HTTPException(status_code=409, detail="Revoke the device first, so its key stops working.")
    d.delete()


@router.get("/fleet/devices/{device_id}/health")
def device_health(device_id: UUID, hours: int = Query(24, ge=1, le=168), ctx=READ):
    from fleet.models import HealthSample
    d = _device(ctx, device_id)
    since = timezone.now() - timedelta(hours=hours)
    return [{"at": s.at.isoformat(), **(s.metrics or {})}
            for s in HealthSample.objects.filter(device=d, at__gte=since).order_by("at")]


# ── Cameras ──────────────────────────────────────────────────────────────────

def _camera(ctx, camera_id):
    from fleet.models import Camera
    c = Camera.objects.filter(camera_id=camera_id, device__organization=ctx.organization).first()
    if c is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return c


def _camera_out(c):
    return {"camera_id": str(c.camera_id), "name": c.name, "position": c.position, "order": c.order, "settings": c.settings or {}}


@router.post("/fleet/devices/{device_id}/cameras", status_code=201)
def add_camera(device_id: UUID, body: CameraIn, ctx=ADMIN):
    from fleet.models import Camera
    d = _device(ctx, device_id)
    try:
        c = Camera.objects.create(device=d, name=body.name.strip(), position=body.position.strip(), order=body.order, settings=body.settings)
    except IntegrityError:
        raise HTTPException(status_code=409, detail="This device already has a camera with that name.")
    return _camera_out(c)


@router.patch("/fleet/cameras/{camera_id}")
def update_camera(camera_id: UUID, body: CameraPatch, ctx=ADMIN):
    c = _camera(ctx, camera_id)
    for field in ("name", "position", "order", "settings"):
        value = getattr(body, field)
        if value is not None:
            setattr(c, field, value.strip() if isinstance(value, str) else value)
    try:
        c.save()
    except IntegrityError:
        raise HTTPException(status_code=409, detail="This device already has a camera with that name.")
    return _camera_out(c)


@router.delete("/fleet/cameras/{camera_id}", status_code=204)
def delete_camera(camera_id: UUID, ctx=ADMIN):
    _camera(ctx, camera_id).delete()
