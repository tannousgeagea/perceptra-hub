"""
Field samples: images devices keep (low confidence, a class of interest, or
a few random ones an hour), with what the model found in them.

    device  POST /edge/samples            what it has → sample id + where to upload
            PUT  <upload url>             straight to storage when it can mint
                                          upload links, else /edge/samples/{id}/file
            POST /edge/samples/{id}/done  → registered into the model's project

Registered through the ingest service: an image tagged field / site / device /
reason, linked to the project, with the device's detections as "prediction"
labels. Active learning ranks it like any unreviewed prediction; people who
edit or review it own it (ingest never overwrites human work).

Unreviewed samples that aren't in an open job are deleted after
FLEET_SAMPLE_RETENTION_DAYS (30), image and file included.
"""
import io
import logging
import os
from datetime import timedelta
from typing import Optional

from django.utils import timezone

from fleet import artifacts
from fleet.models import Device, DeviceModel, FieldSample, SamplingPolicy

logger = logging.getLogger(__name__)

MAX_BYTES = int(os.getenv("FLEET_SAMPLE_MAX_BYTES", str(15 * 1024 * 1024)))
RETENTION_DAYS = int(os.getenv("FLEET_SAMPLE_RETENTION_DAYS", "30"))
UPLOAD_LINK_SECONDS = 15 * 60
FORMATS = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
REVIEWED_STATUSES = ("reviewed", "approved", "dataset")
OPEN_JOB_STATUSES = ("unassigned", "assigned", "in_review")
FIELD_TAG = "field"


class SampleRefused(Exception):
    """The device may not send this sample now; status is the HTTP answer (403 policy, 429 cap…)."""
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# ── policy ───────────────────────────────────────────────────────────────────

def policy_of(model) -> dict:
    p = SamplingPolicy.objects.filter(model=model).first()
    if p is None:
        return {"enabled": False, "low_confidence_below": 0.4, "classes": [], "random_per_hour": 2, "max_per_hour": 10}
    return {"enabled": p.enabled, "low_confidence_below": p.low_confidence_below, "classes": list(p.classes or []),
            "random_per_hour": p.random_per_hour, "max_per_hour": p.max_per_hour}


def device_policy(device: Device, model, policy: Optional[dict] = None) -> dict:
    """What this device should keep for this model: the model's policy, off where the site refuses images."""
    policy = dict(policy or policy_of(model))
    if device.site_id and not device.site.allow_images:
        policy["enabled"] = False
        policy["off_reason"] = "site"
    return policy


# ── storage ──────────────────────────────────────────────────────────────────

def _profile_for(organization):
    from storage.services import get_default_storage_profile
    profile = get_default_storage_profile(organization)
    if profile is None:
        raise SampleRefused(409, "The organization has no default storage to keep field images in.")
    return profile


def _adapter(profile):
    from storage.services import get_storage_adapter_for_profile
    return get_storage_adapter_for_profile(profile)


def _direct_upload(profile, adapter) -> bool:
    """Can the device PUT straight into storage? Not for local disks (their "links" are file:// paths)."""
    if profile.backend == "local" or adapter.read_only:
        return False
    try:
        return bool(adapter.can_presign())
    except Exception:
        return False


# ── the device side ──────────────────────────────────────────────────────────

def create(device: Device, body: dict, *, request_base: str = "") -> tuple:
    """Accept a sample's description; returns (sample, upload instructions)."""
    from ml_models.models import Model, ModelVersion

    model = Model.objects.select_related("project").filter(model_id=body["model_id"], organization_id=device.organization_id,
                                                           is_deleted=False).first()
    if model is None or not DeviceModel.objects.filter(device=device, model=model).exists():
        raise SampleRefused(404, "This device doesn't run that model.")
    policy = device_policy(device, model)
    if not policy["enabled"]:
        raise SampleRefused(403, "Image samples are off for this site." if policy.get("off_reason") == "site"
                            else "Image samples are off for this model.")
    fmt = (body.get("format") or "jpg").lower()
    if fmt not in FORMATS:
        raise SampleRefused(400, f"format must be one of {sorted(FORMATS)}")
    size = int(body.get("size_bytes") or 0)
    if not 0 < size <= MAX_BYTES:
        raise SampleRefused(413, f"An image must be between 1 byte and {MAX_BYTES // (1024 * 1024)} MB.")
    hour_ago = timezone.now() - timedelta(hours=1)
    if FieldSample.objects.filter(device=device, model=model, created_at__gte=hour_ago).count() >= policy["max_per_hour"]:
        raise SampleRefused(429, f"This device already sent {policy['max_per_hour']} samples of this model in the last hour.")

    profile = _profile_for(device.organization)
    version = ModelVersion.objects.filter(model=model, version_id=body.get("version_id")).first() if body.get("version_id") else None
    now = timezone.now()
    captured = body.get("captured_at")
    if captured is None or timezone.is_naive(captured) or captured > now + timedelta(minutes=5):
        captured = now  # a device clock without a zone, or ahead of ours
    sample = FieldSample(organization_id=device.organization_id, device=device, site_id=device.site_id, model=model,
                         version=version, camera=(body.get("camera") or "")[:80], captured_at=captured,
                         reason=body.get("reason") or FieldSample.Reason.RANDOM, predictions=body.get("predictions") or [],
                         storage_profile=profile, file_format=fmt, size_bytes=size,
                         width=body.get("width") or None, height=body.get("height") or None)
    sample.storage_key = f"field/{model.model_id}/{captured:%Y/%m/%d}/{sample.sample_id}.{fmt}"
    sample.save()

    adapter = _adapter(profile)
    if _direct_upload(profile, adapter):
        link = adapter.generate_presigned_url(sample.storage_key, expiration=UPLOAD_LINK_SECONDS, method="PUT")
        upload = {"method": "PUT", "url": link.url, "headers": {"Content-Type": FORMATS[fmt]}, "device_key": False,
                  "expires_at": (timezone.now() + timedelta(seconds=UPLOAD_LINK_SECONDS)).isoformat()}
    else:
        base = artifacts.public_base_url(request_base)
        upload = {"method": "PUT", "url": f"{base}/api/v1/edge/samples/{sample.sample_id}/file",
                  "headers": {"Content-Type": FORMATS[fmt]}, "device_key": True, "expires_at": None}
    return sample, upload


def store(sample: FieldSample, data: bytes) -> None:
    """The upload through the API (storage that can't take a direct upload)."""
    if sample.status != FieldSample.Status.UPLOADING:
        raise SampleRefused(409, "This sample is already complete.")
    if not data or len(data) > MAX_BYTES:
        raise SampleRefused(413, f"An image must be between 1 byte and {MAX_BYTES // (1024 * 1024)} MB.")
    size = _image_size(data)
    if size is None:
        raise SampleRefused(415, "That isn't a readable image.")
    _adapter(sample.storage_profile).upload_file(io.BytesIO(data), sample.storage_key, content_type=FORMATS[sample.file_format])
    sample.size_bytes, (sample.width, sample.height) = len(data), size
    FieldSample.objects.filter(pk=sample.pk).update(size_bytes=sample.size_bytes, width=sample.width, height=sample.height)


def _image_size(data: bytes) -> Optional[tuple]:
    """(width, height) from the image header, or None if it isn't an image."""
    try:
        from PIL import Image as PILImage
        with PILImage.open(io.BytesIO(data)) as im:
            return im.size
    except Exception:
        return None


def _annotations(sample: FieldSample) -> list:
    """The device's detections as prediction labels (those with a box; the rest stay on the sample)."""
    out = []
    vname = sample.version.version_name if sample.version_id else None
    for i, p in enumerate(sample.predictions or []):
        if not isinstance(p, dict):
            continue
        box = p.get("bbox_xyxyn", p.get("bbox"))
        name = p.get("class", p.get("label", p.get("class_name")))
        if not box or name is None:
            continue
        out.append({"external_id": f"p{i}", "class_name": str(name), "bbox_xyxyn": box, "confidence": p.get("confidence", p.get("score")),
                    "source": "prediction", "model_version": vname})
    return out


def finish(sample: FieldSample) -> FieldSample:
    """The file is in storage: register it into the model's project."""
    from ingest.services import IngestOptions, register_items

    if sample.status == FieldSample.Status.REGISTERED:
        return sample
    if not _adapter(sample.storage_profile).file_exists(sample.storage_key):
        raise SampleRefused(409, "The file isn't in storage yet; upload it first.")
    device, site = sample.device, sample.site
    where = site.name if site else device.name
    item = {
        "storage_profile_id": str(sample.storage_profile.storage_profile_id),
        "storage_key": sample.storage_key,
        "external_id": f"field:{sample.sample_id}",
        "name": f"{where} {sample.camera + ' ' if sample.camera else ''}{sample.captured_at:%Y-%m-%d %H:%M:%S}.{sample.file_format}"[:255],
        "file_size": sample.size_bytes,
        **({"width": sample.width, "height": sample.height} if sample.width and sample.height else {}),
        "captured_at": sample.captured_at.isoformat(),
        "source": f"edge:{device.name}"[:255],
        "tags": [t for t in (FIELD_TAG, f"site:{site.name}" if site else None, f"device:{device.name}", f"reason:{sample.reason}") if t],
        "metadata": {"field_sample": str(sample.sample_id), "device": device.name, "site": site.name if site else None,
                     "camera": sample.camera or None, "reason": sample.reason,
                     "model_version": sample.version.version_name if sample.version_id else None},
        "annotations": _annotations(sample),
    }
    result = register_items(sample.organization, None, [item], project=sample.model.project,
                            options=IngestOptions(), source_label="edge")[0]
    if result["status"] == "error":
        sample.status, sample.error = FieldSample.Status.FAILED, (result.get("error") or "")[:500]
        sample.save(update_fields=["status", "error"])
        raise SampleRefused(422, f"Couldn't register the sample: {sample.error}")
    from images.models import Image
    sample.image = Image.objects.filter(image_id=result["image_id"]).first()
    sample.project_image_id = result.get("project_image_id")
    sample.status, sample.registered_at = FieldSample.Status.REGISTERED, timezone.now()
    sample.error = "; ".join(e["error"] for e in (result.get("annotations") or {}).get("errors", []))[:500]
    sample.save(update_fields=["image", "project_image", "status", "registered_at", "error"])
    return sample


# ── cleanup ──────────────────────────────────────────────────────────────────

def _keep(sample: FieldSample) -> bool:
    """Reviewed, in an open job, or touched by a person: it's somebody's work now."""
    from annotations.models import Annotation
    from ingest.services import human_edited
    from jobs.models import JobImage

    pi = sample.project_image
    if pi is None:
        return False
    if pi.reviewed or pi.status in REVIEWED_STATUSES:
        return True
    if JobImage.objects.filter(project_image=pi, is_active=True, job__is_active=True, job__status__in=OPEN_JOB_STATUSES).exists():
        return True
    return any(human_edited(a) for a in Annotation.objects.filter(project_image=pi))


def prune(now=None) -> dict:
    """Delete unreviewed samples past retention (image and file too) and uploads never finished."""
    from images.models import Image
    from projects.models import ProjectImage

    now = now or timezone.now()
    deleted = kept = abandoned = 0
    for s in FieldSample.objects.filter(status=FieldSample.Status.REGISTERED, captured_at__lt=now - timedelta(days=RETENTION_DAYS)) \
            .select_related("project_image", "storage_profile"):
        if _keep(s):
            kept += 1
            continue
        image_pk = s.image_id
        if s.project_image_id:
            ProjectImage.objects.filter(pk=s.project_image_id).delete()  # its annotations go with it
        if image_pk and not ProjectImage.objects.filter(image_id=image_pk).exists():
            Image.objects.filter(pk=image_pk).delete()
        _delete_file(s)
        s.delete()
        deleted += 1
    for s in FieldSample.objects.filter(status__in=(FieldSample.Status.UPLOADING, FieldSample.Status.FAILED),
                                        created_at__lt=now - timedelta(days=1)).select_related("storage_profile"):
        _delete_file(s)
        s.delete()
        abandoned += 1
    return {"deleted": deleted, "kept": kept, "abandoned": abandoned}


def _delete_file(sample: FieldSample) -> None:
    try:
        adapter = _adapter(sample.storage_profile)
        if adapter.file_exists(sample.storage_key):
            adapter.delete_file(sample.storage_key)
    except Exception:
        logger.warning("field sample %s: couldn't delete %s", sample.sample_id, sample.storage_key, exc_info=True)


# ── for the console ──────────────────────────────────────────────────────────

def serialize(s: FieldSample, *, thumbnail: bool = True) -> dict:
    url = None
    if thumbnail and s.image_id:
        try:
            url = s.image.get_download_url(expiration=3600)
        except Exception:  # unreachable storage: the list still works without a picture
            url = None
    pi = s.project_image
    return {
        "sample_id": str(s.sample_id), "reason": s.reason, "status": s.status, "error": s.error or None,
        "captured_at": s.captured_at.isoformat(), "camera": s.camera or None,
        "device": {"device_id": str(s.device.device_id), "name": s.device.name},
        "site": {"site_id": str(s.site.site_id), "name": s.site.name} if s.site_id else None,
        "version_name": (s.version.version_name or f"v{s.version.version_number}") if s.version_id else None,
        "predictions": s.predictions or [],
        "lowest_confidence": min((p.get("confidence") for p in (s.predictions or []) if isinstance(p, dict) and isinstance(p.get("confidence"), (int, float))), default=None),
        "image_id": s.image.image_id if s.image_id else None,
        "project_image_id": s.project_image_id,
        "reviewed": bool(pi and (pi.reviewed or pi.status in REVIEWED_STATUSES)),
        "thumbnail_url": url,
    }
