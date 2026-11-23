"""
The device protocol (version 1). Everything a device ever calls:

    POST /edge/enroll       one-time code → device key          (no other auth)
    POST /edge/heartbeat    report + get desired state          (X-Device-Key)
    POST /edge/rotate-key   swap the key for a new one          (X-Device-Key)

Devices only make outgoing HTTPS calls, so they work behind NAT, proxies and
plant firewalls. The OpenAPI schema of these three routes is the contract
anyone can implement in any language; edge-agent/ is the reference client.
"""
import ipaddress
import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
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


class HeartbeatIn(BaseModel):
    protocol: int = 1
    agent_version: str = Field("", max_length=40)
    accepts: Optional[List[str]] = Field(None, max_length=10)
    hardware: Optional[Dict[str, Any]] = Field(None, description="Send on start or when it changes")
    health: Optional[Dict[str, Any]] = Field(None, description="fps, latency_ms, cpu_percent, gpu_percent, memory_percent, temperature_c, disk_free_gb, errors…")
    health_backlog: List[HealthPoint] = Field(default_factory=list, max_length=120,
                                              description="Reports collected while offline, oldest first; sent in batches")
    models: List[ModelReport] = Field(default_factory=list, max_length=50)
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
