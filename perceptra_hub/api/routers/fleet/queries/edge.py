"""
The device protocol (version 1). Everything a device ever calls:

    POST /edge/enroll       one-time code → device key          (no other auth)
    POST /edge/heartbeat    report + get desired state          (X-Device-Key)
    POST /edge/rotate-key   swap the key for a new one          (X-Device-Key)
    POST /edge/samples      describe a kept image → upload link (X-Device-Key)
    PUT  /edge/samples/{id}/file   the image, when storage takes no direct upload
    POST /edge/samples/{id}/done   uploaded: into the model's project

Devices only make outgoing HTTPS calls, so they work behind NAT, proxies and
plant firewalls. The OpenAPI schema of these three routes is the contract
anyone can implement in any language; edge-agent/ is the reference client.
"""
import ipaddress
import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

router = APIRouter()

MAX_JSON_BYTES = 16_384  # health and hardware reports are small; refuse junk


def _small(value):
    if value is not None and len(json.dumps(value, default=str)) > MAX_JSON_BYTES:
        raise ValueError(f"must be under {MAX_JSON_BYTES} bytes as JSON")
    return value


class EnrollIn(BaseModel):
    code: str = Field(..., min_length=4, max_length=40, description="The one-time code shown in Fleet, dashes optional")
    agent_version: str = Field("", max_length=40)
    hardware: Optional[Dict[str, Any]] = None
    accepts: Optional[List[str]] = Field(None, max_length=10, description="Model formats the device can load, best first")

    @field_validator("hardware")
    @classmethod
    def _limit(cls, v):
        return _small(v)


class EnrollOut(BaseModel):
    device_id: str
    device_key: str = Field(..., description="Shown once. Send it as X-Device-Key on every call.")
    name: str
    site: Optional[str]
    poll_seconds: int
    protocol: int


class ModelReport(BaseModel):
    model_id: str = Field(..., max_length=255)
    version_id: Optional[str] = Field(None, max_length=255, description="The version running now (None if none)")
    status: Literal["pending", "downloading", "running", "failed", "rolled_back"] = "running"
    failed_version_id: Optional[str] = Field(None, max_length=255, description="The version that failed, when the device went back")
    error: Optional[str] = Field(None, max_length=500)


class HealthPoint(BaseModel):
    at: datetime
    metrics: Dict[str, Any]

    @field_validator("metrics")
    @classmethod
    def _limit(cls, v):
        return _small(v)


class PredictionSummary(BaseModel):
    """One hour of one model on one camera, summarised on the device. The current hour is re-sent as it grows."""
    hour: datetime = Field(..., description="Start of the hour (UTC)")
    model_id: str = Field(..., max_length=255)
    version_id: Optional[str] = Field(None, max_length=255)
    camera: str = Field("", max_length=80)
    inferences: int = Field(..., ge=0, le=100_000_000)
    objects: int = Field(0, ge=0, le=1_000_000_000)
    confidence_hist: List[int] = Field(default_factory=list, max_length=10, description="Objects per confidence tenth, [0, .1) … [.9, 1]")
    classes: Dict[str, int] = Field(default_factory=dict, description="Objects per class (up to 100 classes)")

    @field_validator("classes")
    @classmethod
    def _classes(cls, v):
        if len(v) > 101:
            raise ValueError("at most 100 classes per summary")
        return {str(k)[:80]: max(0, int(n)) for k, n in v.items()}


class HeartbeatIn(BaseModel):
    protocol: int = 1
    agent_version: str = Field("", max_length=40)
    accepts: Optional[List[str]] = Field(None, max_length=10)
    hardware: Optional[Dict[str, Any]] = Field(None, description="Send on start or when it changes")
    health: Optional[Dict[str, Any]] = Field(None, description="fps, latency_ms, cpu_percent, gpu_percent, memory_percent, temperature_c, disk_free_gb, errors…")
    health_backlog: List[HealthPoint] = Field(default_factory=list, max_length=120,
                                              description="Reports collected while offline, oldest first; sent in batches")
    models: List[ModelReport] = Field(default_factory=list, max_length=50)
    prediction_stats: List[PredictionSummary] = Field(default_factory=list, max_length=48,
                                                      description="Hourly prediction summaries (POST /v1/predictions on the agent)")
    state_token: Optional[str] = Field(None, max_length=64, description="From the last answer; an unchanged state then gets a tiny reply")
    refresh: bool = Field(False, description="Ask for fresh download links even if nothing changed")

    @field_validator("hardware", "health")
    @classmethod
    def _limit(cls, v):
        return _small(v)


def client_ip(request: Request) -> Optional[str]:
    raw = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() or request.headers.get("x-real-ip") \
        or (request.client.host if request.client else None)
    try:
        return str(ipaddress.ip_address(raw)) if raw else None
    except ValueError:
        return None


def authenticate_device(x_device_key: Optional[str] = Header(None, description="The device key from /edge/enroll")):
    from fleet import services

    device = services.authenticate(x_device_key or "")
    if device is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Unknown or revoked device key. Enroll the device again with a new code from Fleet.")
    return device


def _base(request: Request) -> str:
    return str(request.base_url)


@router.post("/edge/enroll", response_model=EnrollOut)
def enroll(body: EnrollIn, request: Request):
    from fleet import services

    try:
        device, key = services.enroll(body.code, agent_version=body.agent_version, hardware=body.hardware,
                                      accepts=body.accepts, ip=client_ip(request))
    except services.FleetError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return EnrollOut(device_id=str(device.device_id), device_key=key, name=device.name,
                     site=device.site.name if device.site_id else None,
                     poll_seconds=services.poll_seconds(device), protocol=services.PROTOCOL)


@router.post("/edge/heartbeat")
def heartbeat(body: HeartbeatIn, request: Request, device=Depends(authenticate_device)):
    from fleet import services

    return services.heartbeat(device, body.model_dump(), ip=client_ip(request), request_base=_base(request))


@router.post("/edge/rotate-key")
def rotate_key(device=Depends(authenticate_device)):
    from fleet import services

    return {"device_key": services.rotate_key(device)}


# ── field samples ────────────────────────────────────────────────────────────

class SamplePrediction(BaseModel):
    class_: str = Field(..., alias="class", max_length=120)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    bbox_xyxyn: Optional[List[float]] = Field(None, min_length=4, max_length=4, description="[x1, y1, x2, y2], 0-1")

    model_config = {"populate_by_name": True}


class SampleIn(BaseModel):
    model_id: str = Field(..., max_length=255)
    version_id: Optional[str] = Field(None, max_length=255)
    camera: str = Field("", max_length=80)
    captured_at: Optional[datetime] = None
    reason: Literal["low_confidence", "class", "random"]
    format: Literal["jpg", "jpeg", "png", "webp"] = "jpg"
    size_bytes: int = Field(..., gt=0)
    width: Optional[int] = Field(None, gt=0, le=100_000)
    height: Optional[int] = Field(None, gt=0, le=100_000)
    predictions: List[SamplePrediction] = Field(default_factory=list, max_length=500)


def _refused(e):
    return HTTPException(status_code=e.status, detail=str(e))


def _own_sample(device, sample_id):
    from fleet.models import FieldSample
    s = (FieldSample.objects.select_related("storage_profile", "device__site", "site", "model__project", "version", "organization")
         .filter(sample_id=sample_id, device=device).first())
    if s is None:
        raise HTTPException(status_code=404, detail="No such sample from this device")
    return s


@router.post("/edge/samples", status_code=201)
def create_sample(body: SampleIn, request: Request, device=Depends(authenticate_device)):
    """Describe an image the device kept; the answer says where to upload it (then call …/done)."""
    from fleet import samples

    payload = body.model_dump()
    payload["predictions"] = [{"class": p.class_, "confidence": p.confidence, "bbox_xyxyn": p.bbox_xyxyn} for p in body.predictions]
    try:
        sample, upload = samples.create(device, payload, request_base=_base(request))
    except samples.SampleRefused as e:
        raise _refused(e)
    return {"sample_id": str(sample.sample_id), "upload": upload}


@router.put("/edge/samples/{sample_id}/file")
def upload_sample_file(sample_id: UUID, data: bytes = Body(..., media_type="application/octet-stream"),
                       content_length: Optional[int] = Header(None), device=Depends(authenticate_device)):
    """The image itself, for storage that can't take a direct upload."""
    from fleet import samples

    if content_length and content_length > samples.MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        samples.store(_own_sample(device, sample_id), data)
    except samples.SampleRefused as e:
        raise _refused(e)
    return {"stored": True}


@router.post("/edge/samples/{sample_id}/done")
def finish_sample(sample_id: UUID, device=Depends(authenticate_device)):
    """The file is uploaded: put the sample into the model's project."""
    from fleet import samples

    try:
        s = samples.finish(_own_sample(device, sample_id))
    except samples.SampleRefused as e:
        raise _refused(e)
    return {"sample_id": str(s.sample_id), "status": s.status, "image_id": s.image.image_id if s.image_id else None}
