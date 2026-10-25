"""
Register images by reference — the producer side of Perceptra.

An external system (a production platform, a data lake, a labelling vendor)
pushes images that already live in a storage Perceptra can read — typically
an external read-only StorageProfile — together with their metadata, tags and
annotations. Nothing is downloaded or copied: an image is its
(storage_profile, storage_key), URLs are built from the profile on demand.

Idempotency
    image       by (organization, external_id), else by (storage_profile, storage_key)
    annotation  by (project_image, external_id)
Re-registering the same payload is "unchanged"; a changed one updates in place.

Human work always wins
    Every ingested annotation remembers what was last written
    (meta["ingested"]: class + box). If a person has since edited, reviewed or
    deleted it, a re-push never overwrites it: the annotation is reported as a
    conflict and left alone.

Label provenance
    source "prediction" -> annotation_source "prediction" (model output to review)
    source "verified"   -> annotation_source "imported", reviewed: ground truth
                           confirmed in the producing system; exported like
                           manual labels, never audited as a missed detection.

Used by POST /api/v1/ingest/images (api/routers/ingest) and its SDK client.
"""
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from django.db import transaction
from django.utils import timezone

logger = logging.getLogger(__name__)

SOURCE_PREDICTION = "prediction"
SOURCE_VERIFIED = "verified"
ANNOTATION_SOURCE = {SOURCE_PREDICTION: "prediction", SOURCE_VERIFIED: "imported"}

STATUS_CREATED = "created"
STATUS_UPDATED = "updated"
STATUS_UNCHANGED = "unchanged"
STATUS_ERROR = "error"

BOX_TYPE = "box"
POLYGON_TYPE = "polygon"
CLASS_COLORS = (
    "#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899",
    "#14b8a6", "#84cc16", "#f43f5e", "#6366f1",
)
ROUND = 6


class IngestError(Exception):
    """A single item cannot be registered (reported per item, never aborts the batch)."""


@dataclass
class IngestOptions:
    auto_assign_job: bool = False
    batch_id: Optional[str] = None
    create_missing_classes: bool = False
    verify_exists: bool = False


@dataclass
class _BatchCache:
    profiles: Dict[str, Any] = field(default_factory=dict)
    classes: Optional[Dict[str, Any]] = None
    types: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


def _bbox(values) -> List[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise IngestError("bbox_xyxyn must be [x1, y1, x2, y2] normalized to 0-1")
    try:
        x1, y1, x2, y2 = [min(1.0, max(0.0, float(v))) for v in values]
    except (TypeError, ValueError):
        raise IngestError("bbox_xyxyn must be numeric")
    if x2 <= x1 or y2 <= y1:
        raise IngestError("bbox_xyxyn must have x2 > x1 and y2 > y1")
    return [round(v, ROUND) for v in (x1, y1, x2, y2)]


def _polygon(values) -> Optional[List[List[float]]]:
    if not values:
        return None
    try:
        points = [[round(min(1.0, max(0.0, float(x))), ROUND), round(min(1.0, max(0.0, float(y))), ROUND)]
                  for x, y in values]
    except (TypeError, ValueError):
        raise IngestError("polygon_xyn must be [[x, y], ...] normalized to 0-1")
    if len(points) < 3:
        raise IngestError("polygon_xyn needs at least 3 points")
    return points


def _snapshot(annotation_class_id: int, data, polygon) -> dict:
    return {
        "class_id": annotation_class_id,
        "data": [round(float(v), ROUND) for v in (data or [])],
        "polygon": polygon,
    }


def human_edited(annotation) -> bool:
    """A person changed, reviewed or deleted this annotation after it was ingested."""
    ingested = (annotation.meta or {}).get("ingested")
    if ingested is None:
        return True  # not written by ingestion
    if annotation.reviewed_by_id is not None or annotation.deleted_by_id is not None:
        return True
    # Review endpoints flag reviewed without always recording who: a reviewed
    # prediction was reviewed by a person (verified imports arrive reviewed).
    if annotation.reviewed and (annotation.meta or {}).get("source") != SOURCE_VERIFIED:
        return True
    if annotation.edit_type not in (None, "", "none"):
        return True
    current = _snapshot(annotation.annotation_class_id, annotation.data, annotation.polygon_data)
    return current != ingested


# ── Resolution ────────────────────────────────────────────────────────────

def _resolve_profile(organization, item: dict, cache: _BatchCache):
    from storage.models import StorageProfile

    ref = item.get("storage_profile_id") or item.get("storage_profile_name")
    if not ref:
        raise IngestError("storage_profile_id or storage_profile_name is required")
    ref = str(ref)
    if ref in cache.profiles:
        return cache.profiles[ref]
    qs = StorageProfile.objects.filter(organization=organization, is_active=True)
    try:
        profile = qs.filter(storage_profile_id=uuid.UUID(ref)).first()
    except ValueError:
        profile = qs.filter(name=ref).first()
    if profile is None:
        raise IngestError(f"storage profile '{ref}' not found or inactive")
    cache.profiles[ref] = profile
    return profile


def _class_map(project, cache: _BatchCache) -> Dict[str, Any]:
    from annotations.models import AnnotationClass

    if cache.classes is None:
        cache.classes = {}
        for cls in AnnotationClass.objects.filter(annotation_group__project=project).order_by("id"):
            cache.classes.setdefault(cls.name.strip().lower(), cls)
    return cache.classes


def _resolve_class(project, name: str, options: IngestOptions, cache: _BatchCache):
    from annotations.models import AnnotationClass, AnnotationGroup

    if not name or not str(name).strip():
        raise IngestError("class_name is required")
    classes = _class_map(project, cache)
    key = str(name).strip().lower()
    if key in classes:
        return classes[key]
    if not options.create_missing_classes:
        raise IngestError(f"class '{name}' does not exist in the project (create_missing_classes is off)")
    group = AnnotationGroup.objects.filter(project=project).order_by("id").first()
    if group is None:
        group = AnnotationGroup.objects.create(project=project, name="default")
    next_id = (AnnotationClass.objects.filter(annotation_group=group).order_by("-class_id")
               .values_list("class_id", flat=True).first())
    next_id = 0 if next_id is None else next_id + 1
    cls = AnnotationClass.objects.create(
        annotation_group=group, class_id=next_id, name=str(name).strip(),
        color=CLASS_COLORS[next_id % len(CLASS_COLORS)],
    )
    classes[key] = cls
    return cls


def _annotation_type(name: str, cache: _BatchCache):
    from annotations.models import AnnotationType

    if name not in cache.types:
        cache.types[name] = AnnotationType.objects.get_or_create(name=name)[0]
    return cache.types[name]


def _tag(organization, name: str, cache: _BatchCache):
    from images.models import Tag

    if name not in cache.tags:
        cache.tags[name] = Tag.objects.get_or_create(organization=organization, name=name[:100])[0]
    return cache.tags[name]


# ── Image ─────────────────────────────────────────────────────────────────

IMAGE_FIELDS = ("storage_profile_id", "storage_key", "width", "height", "file_size", "file_format",
                "checksum", "meta_info", "source_of_origin")


def _upsert_image(organization, user, item: dict, profile, source_label: str):
    from images.models import Image

    key = (item.get("storage_key") or "").strip().lstrip("/")
    if not key:
        raise IngestError("storage_key is required")
    try:
        width, height = int(item.get("width") or 0), int(item.get("height") or 0)
    except (TypeError, ValueError):
        raise IngestError("width and height must be integers")
    if width <= 0 or height <= 0:
        raise IngestError("width and height (pixels) are required: images are never downloaded")

    external_id = item.get("external_id")
    image = None
    if external_id:
        image = Image.objects.filter(organization=organization, external_id=external_id).first()
    if image is None:
        image = Image.objects.filter(storage_profile=profile, storage_key=key).first()
        if image is not None and image.external_id and external_id and image.external_id != external_id:
            raise IngestError(f"storage key already registered under external_id '{image.external_id}'")

    filename = os.path.basename(key)
    extension = os.path.splitext(filename)[1].lstrip(".").lower()[:10] or "jpg"
    meta = dict(item.get("metadata") or {})
    if item.get("captured_at"):
        meta["captured_at"] = item["captured_at"]
    meta.pop("_ingest", None)
    if image is not None and (image.meta_info or {}).get("_ingest"):
        meta["_ingest"] = image.meta_info["_ingest"]  # bookkeeping of _sync_tags
    values = {
        "storage_profile_id": profile.id,
        "storage_key": key,
        "width": width,
        "height": height,
        "file_size": int(item.get("file_size") or 0),
        "file_format": extension,
        "checksum": item.get("checksum") or None,
        "meta_info": meta,
        "source_of_origin": (item.get("source") or source_label or "external")[:255],
    }

    if image is None:
        image = Image.objects.create(
            image_id=str(uuid.uuid4()),
            organization=organization,
            external_id=external_id,
            name=(item.get("name") or filename)[:255],
            original_filename=filename[:255],
            image_name=(item.get("name") or filename)[:255],
            uploaded_by=user,
            **values,
        )
        return image, STATUS_CREATED

    changed = [f for f in IMAGE_FIELDS if getattr(image, f) != values[f]]
    if external_id and image.external_id != external_id:
        image.external_id = external_id
        changed.append("external_id")
    if not changed:
        return image, STATUS_UNCHANGED
    for f in IMAGE_FIELDS:
        setattr(image, f, values[f])
    image.save(update_fields=[*{*changed, *IMAGE_FIELDS}, "updated_at"])
    return image, STATUS_UPDATED


def _sync_tags(organization, user, image, tag_names: List[str], cache: _BatchCache) -> bool:
    """The image's ingested tags become exactly `tag_names`; tags people added stay."""
    from images.models import ImageTag

    wanted = {t.strip()[:100] for t in tag_names or [] if t and t.strip()}
    previous = set(((image.meta_info or {}).get("_ingest") or {}).get("tags") or [])
    changed = False
    for name in wanted - previous:
        _, created = ImageTag.objects.get_or_create(image=image, tag=_tag(organization, name, cache),
                                                    defaults={"tagged_by": user})
        changed |= created
    stale = previous - wanted
    if stale:
        changed |= ImageTag.objects.filter(image=image, tag__name__in=stale).delete()[0] > 0
    if wanted != previous:
        meta = dict(image.meta_info or {})
        meta["_ingest"] = {**(meta.get("_ingest") or {}), "tags": sorted(wanted)}
        type(image).objects.filter(pk=image.pk).update(meta_info=meta)
        image.meta_info = meta
    return changed


# ── Project link + annotations ────────────────────────────────────────────

def _link_project(project, image, user, item: dict, options: IngestOptions):
    from projects.models import ProjectImage

    pi, created = ProjectImage.objects.get_or_create(
        project=project, image=image,
        defaults={
            "added_by": user,
            "priority": int(item.get("review_priority") or 0),
            "marked_as_null": bool(item.get("mark_as_null")),
            "job_assignment_status": "waiting" if options.auto_assign_job else "excluded",
            "metadata": {"ingest": {"external_id": item.get("external_id")}},
        },
    )
    updates = {}
    if not pi.is_active:
        updates["is_active"] = True
    if item.get("review_priority") is not None and pi.priority != int(item["review_priority"]):
        updates["priority"] = int(item["review_priority"])
    if item.get("mark_as_null") is not None and pi.marked_as_null != bool(item["mark_as_null"]):
        updates["marked_as_null"] = bool(item["mark_as_null"])
    if updates:
        ProjectImage.objects.filter(pk=pi.pk).update(**updates, updated_at=timezone.now())
        for k, v in updates.items():
            setattr(pi, k, v)
    if created and options.auto_assign_job:
        from common_utils.jobs.utils import assign_uploaded_image_to_batch

        assign_uploaded_image_to_batch(pi, options.batch_id, user)
    return pi, created or bool(updates)


def _sync_annotations(project, pi, user, payload: List[dict], options: IngestOptions, cache: _BatchCache) -> dict:
    """Upsert the pushed annotations of one project image; soft-delete the
    ingested ones no longer pushed. Human-touched annotations are never changed."""
    from annotations.models import Annotation

    now = timezone.now()
    existing = {
        a.external_id: a for a in Annotation.objects.filter(project_image=pi, external_id__isnull=False)
    }
    counts = {"created": 0, "updated": 0, "unchanged": 0, "removed": 0, "conflicts": [], "errors": []}
    to_create, to_update, seen = [], [], set()

    for raw in payload or []:
        ext_id = str(raw.get("external_id") or "").strip()
        try:
            if not ext_id:
                raise IngestError("annotation external_id is required")
            if ext_id in seen:
                raise IngestError("duplicate annotation external_id in this image")
            seen.add(ext_id)
            source = raw.get("source") or SOURCE_PREDICTION
            if source not in ANNOTATION_SOURCE:
                raise IngestError(f"source must be one of {sorted(ANNOTATION_SOURCE)}")
            cls = _resolve_class(project, raw.get("class_name"), options, cache)
            data = _bbox(raw.get("bbox_xyxyn"))
            polygon = _polygon(raw.get("polygon_xyn"))
            confidence = raw.get("confidence")
            if confidence is not None:
                confidence = min(1.0, max(0.0, float(confidence)))
        except (IngestError, TypeError, ValueError) as e:
            counts["errors"].append({"external_id": ext_id or None, "error": str(e)})
            continue

        snapshot = _snapshot(cls.id, data, polygon)
        verified = source == SOURCE_VERIFIED
        meta = {"source": source, "metadata": raw.get("metadata") or {}, "ingested": snapshot}
        values = {
            "annotation_class": cls,
            "annotation_type": _annotation_type(POLYGON_TYPE if polygon else BOX_TYPE, cache),
            "data": data,
            "polygon_data": polygon,
            "annotation_source": ANNOTATION_SOURCE[source],
            "confidence": confidence,
            "model_version": (raw.get("model_version") or None) and str(raw["model_version"])[:50],
            "meta": meta,
            "reviewed": verified,
            "is_active": True,
            "is_deleted": False,
        }

        current = existing.get(ext_id)
        if current is None:
            to_create.append(Annotation(
                project_image=pi, external_id=ext_id, created_by=user, updated_by=user,
                reviewed_at=now if verified else None, **values,
            ))
            continue
        if human_edited(current):
            counts["conflicts"].append({"external_id": ext_id, "reason": "edited by a person since the last push"})
            continue
        same = (
            current.meta == meta and current.annotation_source == values["annotation_source"]
            and current.confidence == confidence and current.model_version == values["model_version"]
            and current.is_active and not current.is_deleted
        )
        if same:
            counts["unchanged"] += 1
            continue
        for k, v in values.items():
            setattr(current, k, v)
        current.updated_by = user
        current.reviewed_at = (current.reviewed_at or now) if verified else None
        to_update.append(current)

    # Pushed before, not any more: withdraw it (only while nobody touched it).
    for ext_id, current in existing.items():
        if ext_id in seen or current.is_deleted:
            continue
        if human_edited(current):
            counts["conflicts"].append({"external_id": ext_id, "reason": "absent from push but edited by a person"})
            continue
        current.is_deleted, current.is_active, current.deleted_at = True, False, now
        to_update.append(current)
        counts["removed"] += 1

    if to_create:
        Annotation.objects.bulk_create(to_create)
        counts["created"] = len(to_create)
    if to_update:
        Annotation.objects.bulk_update(to_update, [
            "annotation_class", "annotation_type", "data", "polygon_data", "annotation_source", "confidence",
            "model_version", "meta", "reviewed", "reviewed_at", "is_active", "is_deleted", "deleted_at",
            "updated_by",
        ])
        counts["updated"] = len(to_update) - counts["removed"]

    if to_create or to_update:
        _refresh_project_image_flags(pi)
    return counts


def _refresh_project_image_flags(pi) -> None:
    from projects.models import ProjectImage

    active = pi.annotations.filter(is_deleted=False, is_active=True)
    annotated = active.exists()
    updates = {"annotated": annotated}
    if not pi.reviewed and not pi.finalized:
        updates["status"] = "annotated" if annotated else "unannotated"
    ProjectImage.objects.filter(pk=pi.pk).update(**updates, updated_at=timezone.now())


# ── Entry points ──────────────────────────────────────────────────────────

def register_items(organization, user, items: List[dict], project=None,
                   options: Optional[IngestOptions] = None, source_label: str = "") -> List[dict]:
    """Register a batch by reference. One savepoint per item: a bad item is
    reported and the rest of the batch still lands."""
    options = options or IngestOptions()
    cache = _BatchCache()
    results = []
    for index, item in enumerate(items):
        result = {"index": index, "external_id": item.get("external_id"), "status": STATUS_ERROR}
        try:
            with transaction.atomic():
                profile = _resolve_profile(organization, item, cache)
                if options.verify_exists:
                    from storage.services import get_storage_adapter_for_profile

                    key = (item.get("storage_key") or "").strip().lstrip("/")
                    if not get_storage_adapter_for_profile(profile).file_exists(key):
                        raise IngestError(f"'{key}' does not exist in storage profile '{profile.name}'")
                image, status = _upsert_image(organization, user, item, profile, source_label)
                if _sync_tags(organization, user, image, item.get("tags") or [], cache) and status == STATUS_UNCHANGED:
                    status = STATUS_UPDATED
                result.update(image_id=image.image_id, storage_profile_id=str(profile.storage_profile_id))
                if project is not None:
                    pi, linked = _link_project(project, image, user, item, options)
                    result["project_image_id"] = pi.id
                    if linked and status == STATUS_UNCHANGED:
                        status = STATUS_UPDATED
                    if item.get("annotations") is not None:
                        counts = _sync_annotations(project, pi, user, item["annotations"], options, cache)
                        result["annotations"] = counts
                        if (counts["created"] or counts["updated"] or counts["removed"]) and status == STATUS_UNCHANGED:
                            status = STATUS_UPDATED
                elif item.get("annotations"):
                    raise IngestError("annotations need a project_id")
                result["status"] = status
        except IngestError as e:
            result["error"] = str(e)
        except Exception as e:
            logger.exception("ingest item %s failed", index)
            result["error"] = f"{type(e).__name__}: {e}"
        results.append(result)
    return results


def remove_items(organization, external_ids: List[str], project=None) -> List[dict]:
    """Withdraw registered images: deactivate their project links (in
    `project`, or everywhere) and soft-delete their unedited ingested
    annotations. Storage is never touched."""
    from annotations.models import Annotation
    from images.models import Image
    from projects.models import ProjectImage

    now = timezone.now()
    results = []
    for external_id in external_ids:
        image = Image.objects.filter(organization=organization, external_id=external_id).first()
        if image is None:
            results.append({"external_id": external_id, "status": "not_found"})
            continue
        with transaction.atomic():
            links = ProjectImage.objects.filter(image=image, is_active=True)
            if project is not None:
                links = links.filter(project=project)
            link_ids = list(links.values_list("id", flat=True))
            withdrawn = [
                a for a in Annotation.objects.filter(project_image_id__in=link_ids, external_id__isnull=False,
                                                     is_deleted=False)
                if not human_edited(a)
            ]
            for a in withdrawn:
                a.is_deleted, a.is_active, a.deleted_at = True, False, now
            if withdrawn:
                Annotation.objects.bulk_update(withdrawn, ["is_deleted", "is_active", "deleted_at"])
            ProjectImage.objects.filter(id__in=link_ids).update(is_active=False, updated_at=now)
        results.append({"external_id": external_id, "status": "removed", "project_images": len(link_ids),
                        "annotations_removed": len(withdrawn)})
    return results
