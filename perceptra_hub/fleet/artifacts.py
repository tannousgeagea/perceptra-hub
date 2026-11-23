"""
The files devices download, and what comes with them.

Before a device is told to run a version, each of the version's files is
prepared once: its SHA-256 and size (so the device can verify what it
downloaded) and a manifest (classes, input size, task…) so the device's own
inference code doesn't hard-code anything that retraining can change.

Downloads never pass through the API: devices get short-lived signed links
straight to storage.
"""
import hashlib
import logging
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from django.utils import timezone

from fleet.models import ModelArtifact

logger = logging.getLogger(__name__)

# Formats a version can have, and the version field holding each file.
FORMAT_FIELDS = {"onnx": "onnx_model_key", "pt": "checkpoint_key"}
# Used when a device doesn't say what it accepts: the portable one first.
DEFAULT_ACCEPTS = ["onnx", "pt"]
URL_TTL = timedelta(minutes=15)


def available_formats(version) -> Dict[str, str]:
    """{format: storage key} for the files this version has."""
    return {fmt: getattr(version, field) for fmt, field in FORMAT_FIELDS.items() if getattr(version, field, None)}


def pick_format(version, accepts: Optional[List[str]]) -> Optional[str]:
    """The first format in the device's preference order that the version has."""
    have = available_formats(version)
    for fmt in (accepts or DEFAULT_ACCEPTS):
        if fmt in have:
            return fmt
    return None


def class_names(version) -> List[str]:
    """Ordered class names of the model's project (the training label order)."""
    try:
        from annotations.models import AnnotationClass, AnnotationGroup
        group = AnnotationGroup.objects.filter(project=version.model.project).first()
        if group is None:
            return []
        return list(AnnotationClass.objects.filter(annotation_group=group).order_by("class_id").values_list("name", flat=True))
    except Exception as e:  # never block a rollout on a missing manifest field
        logger.warning("fleet: class names for %s: %s", version.version_id, e)
        return []


def build_manifest(version, fmt: str) -> dict:
    model = version.model
    config = version.config or {}
    return {
        "model_id": model.model_id,
        "model_name": model.name,
        "version_id": version.version_id,
        "version_name": version.version_name,
        "version_number": version.version_number,
        "task": getattr(model.task, "name", None),
        "framework": getattr(model.framework, "name", None),
        "format": fmt,
        "classes": class_names(version),
        "input_size": config.get("image_size") or config.get("imgsz"),
        "metrics": version.metrics or {},
    }


def _sha256_of(adapter, key: str) -> Tuple[str, int]:
    """Download to a temp file (models can be large) and hash it in chunks."""
    fd, tmp = tempfile.mkstemp(prefix="fleet-artifact-")
    os.close(fd)
    try:
        adapter.download_file(key, destination=Path(tmp))
        digest, size = hashlib.sha256(), 0
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def prepare_version(version) -> List[ModelArtifact]:
    """Checksum and describe every file of the version (once; prepared ones are kept)."""
    from storage.services import get_storage_adapter_for_profile

    prepared = []
    adapter = None
    for fmt, key in available_formats(version).items():
        art, _ = ModelArtifact.objects.get_or_create(version=version, format=fmt, defaults={"storage_key": key})
        if art.prepared_at and art.storage_key == key and art.sha256:
            prepared.append(art)
            continue
        art.storage_key = key
        try:
            adapter = adapter or get_storage_adapter_for_profile(version.storage_profile)
            art.sha256, art.size_bytes = _sha256_of(adapter, key)
            art.manifest = build_manifest(version, fmt)
            art.error = ""
            art.prepared_at = timezone.now()
        except Exception as e:
            logger.exception("fleet: preparing %s %s failed", version.version_id, fmt)
            art.error = str(e)[:500]
            art.prepared_at = None
        art.save()
        prepared.append(art)
    return prepared


def is_prepared(version) -> bool:
    """Every file the version has is checksummed (for its current storage key)."""
    done = {a.format: a.storage_key for a in ModelArtifact.objects.filter(version=version, prepared_at__isnull=False).exclude(sha256="")}
    return all(done.get(fmt) == key for fmt, key in available_formats(version).items())


def ready_artifact(version, accepts: Optional[List[str]]) -> Optional[ModelArtifact]:
    """The prepared artifact this device should download, or None if not ready yet."""
    ready = {a.format: a for a in ModelArtifact.objects.filter(version=version, prepared_at__isnull=False).exclude(sha256="")}
    for fmt in (accepts or DEFAULT_ACCEPTS):
        if fmt in ready:
            return ready[fmt]
    return None


def public_base_url(request_base: str = "") -> str:
    """Where devices reach this API; FLEET_PUBLIC_BASE_URL wins over the request's own address."""
    return (os.getenv("FLEET_PUBLIC_BASE_URL") or request_base or "").rstrip("/")


def download_url(artifact: ModelArtifact, request_base: str = "") -> Tuple[str, "timezone.datetime"]:
    """A short-lived link straight to the file in storage."""
    from storage.services import get_local_media_url, get_storage_adapter_for_profile

    profile = artifact.version.storage_profile
    expires = timezone.now() + URL_TTL
    if profile.backend == "local":
        url = get_local_media_url(f"{profile.config['base_path']}/{artifact.storage_key}")
        if url.startswith("/"):
            url = public_base_url(request_base) + url
        return url, expires
    adapter = get_storage_adapter_for_profile(profile)
    return adapter.generate_presigned_url(artifact.storage_key, expiration=int(URL_TTL.total_seconds()), method="GET").url, expires
