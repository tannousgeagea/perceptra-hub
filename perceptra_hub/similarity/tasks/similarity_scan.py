"""
similarity/tasks.py
===================
Celery tasks for running perceptual-similarity scans asynchronously.

Architecture
------------
``run_similarity_scan`` is the primary task.  It:

1. Resolves the image queryset from the scan's scope (datalake or project).
2. Hashes each image using ``doppix`` — downloading from storage only when
   no cached ``ImageHash`` row exists.
3. Runs the greedy clustering algorithm from ``doppix``.
4. Persists ``SimilarityCluster`` + ``SimilarityClusterMember`` rows inside a
   single transaction.
5. Updates ``SimilarityScan.status`` / progress counters throughout.

Progress updates are written directly to the DB every ``PROGRESS_BATCH``
images so the polling endpoint always returns fresh data without Redis pub/sub.

Error handling
--------------
Any unhandled exception marks the scan as ``FAILED`` and stores the traceback
in ``error_log``.  The task is NOT retried automatically — users re-run it
manually with new parameters if needed.
"""

import logging
import traceback
from io import BytesIO
from typing import Optional

from celery import shared_task
from django.db import transaction
from django.utils import timezone

logger = logging.getLogger(__name__)

# How often to flush progress to the DB (every N images hashed)
PROGRESS_BATCH = 50


@shared_task(
    bind=True,
    name="similarity:run_scan",
    max_retries=0,          # manual retry only
    acks_late=True,         # only ack after task body finishes
    time_limit=3600,        # hard kill after 1 hour
    soft_time_limit=3540,   # SoftTimeLimitExceeded raised at 59 min
)
def run_similarity_scan(self, scan_id: str) -> dict:
    """
    Execute a similarity scan identified by *scan_id* (UUID string).

    Called by ``POST /similarity/scans`` after the ``SimilarityScan`` row
    is created with status=PENDING.

    Returns a summary dict that Celery stores as the task result.
    """
    from similarity.models import (
        SimilarityScan, ScanStatus, ImageHash, HashAlgorithm,
        SimilarityCluster, SimilarityClusterMember, MemberRole, ClusterStatus,
    )
    from images.models import Image

    # ── 1. Fetch scan & mark running ────────────────────────────────────────
    try:
        scan = SimilarityScan.objects.select_related(
            "organization", "project"
        ).get(scan_id=scan_id)
    except SimilarityScan.DoesNotExist:
        logger.error("Scan %s not found", scan_id)
        return {"error": "scan_not_found"}

    if scan.status == ScanStatus.CANCELLED:
        logger.info("Scan %s was cancelled before it started", scan_id)
        return {"status": "cancelled"}

    scan.status     = ScanStatus.RUNNING
    scan.started_at = timezone.now()
    scan.task_id    = self.request.id
    scan.save(update_fields=["status", "started_at", "task_id", "updated_at"])

    try:
        # ── 2. Resolve image queryset ────────────────────────────────────────
        image_qs = Image.objects.filter(organization=scan.organization)

        if scan.project_id:
            image_qs = image_qs.filter(
                project_assignments__project=scan.project,
                project_assignments__is_active=True,
            )

        images = list(
            image_qs.select_related("storage_profile").values(
                "id", "image_id", "storage_key", "storage_profile_id",
                "storage_profile__backend", "storage_profile__config",
            )
        )

        scan.total_images = len(images)
        scan.save(update_fields=["total_images", "updated_at"])

        if not images:
            _mark_completed(scan, clusters_found=0)
            return {"status": "completed", "clusters": 0, "images": 0}

        # ── 3. Load or compute hashes ────────────────────────────────────────
        algorithm = scan.algorithm

        # Fetch all cached hashes for this org + algorithm in one query
        cached = {
            row["image_id"]: row["hash_value"]
            for row in ImageHash.objects.filter(
                image__organization=scan.organization,
                algorithm=algorithm,
            ).values("image_id", "hash_value")
        }

        hash_map: list[tuple[str, object]] = []   # [(image_id_str, ImageHash)]
        to_cache: list[ImageHash] = []
        hashed = 0

        for img in images:
            # Check for cancellation every batch
            if hashed % PROGRESS_BATCH == 0 and hashed > 0:
                scan.refresh_from_db(fields=["status"])
                if scan.status == ScanStatus.CANCELLED:
                    logger.info("Scan %s cancelled mid-run", scan_id)
                    return {"status": "cancelled"}

                scan.hashed_images = hashed
                scan.save(update_fields=["hashed_images", "updated_at"])

            img_pk   = img["id"]
            img_uuid = str(img["image_id"])

            if img_uuid in cached:
                h = _parse_hash(cached[img_uuid])
                if h is not None:
                    hash_map.append((img_uuid, h))
                    hashed += 1
                    continue

            # Not cached — download from storage and compute
            raw_hash = _compute_hash_from_storage(img, algorithm)
            if raw_hash is None:
                logger.warning("Could not hash image %s — skipping", img_uuid)
                hashed += 1
                continue

            hash_str = str(raw_hash)
            hash_map.append((img_uuid, raw_hash))
            to_cache.append(
                ImageHash(
                    image_id=img_pk,
                    algorithm=algorithm,
                    hash_value=hash_str,
                )
            )
            hashed += 1

        # Bulk-save newly computed hashes
        if to_cache:
            ImageHash.objects.bulk_create(to_cache, ignore_conflicts=True)

        scan.hashed_images = hashed
        scan.save(update_fields=["hashed_images", "updated_at"])

        # ── 4. Greedy clustering (pure Python — no DB I/O) ──────────────────
        raw_clusters = _greedy_cluster(hash_map, threshold=scan.threshold)

        # Only keep clusters with ≥2 members
        raw_clusters = [c for c in raw_clusters if len(c["members"]) >= 2]

        # ── 5. Persist clusters ──────────────────────────────────────────────
        _persist_clusters(scan, raw_clusters)

        # ── 6. Mark completed ────────────────────────────────────────────────
        _mark_completed(scan, clusters_found=len(raw_clusters))

        logger.info(
            "Scan %s completed: %d images, %d clusters",
            scan_id, len(images), len(raw_clusters),
        )
        return {
            "status": "completed",
            "images": len(images),
            "clusters": len(raw_clusters),
        }

    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Scan %s failed: %s", scan_id, exc)
        scan.status    = ScanStatus.FAILED
        scan.error_log = tb
        scan.save(update_fields=["status", "error_log", "updated_at"])
        raise  # re-raise so Celery marks task as FAILURE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mark_completed(scan, clusters_found: int):
    from similarity.models import ScanStatus
    scan.status        = ScanStatus.COMPLETED
    scan.clusters_found = clusters_found
    scan.completed_at  = timezone.now()
    scan.save(update_fields=["status", "clusters_found", "completed_at", "updated_at"])


def _parse_hash(hash_str: str):
    """Re-hydrate an imagehash object from its hex string."""
    try:
        import imagehash
        return imagehash.hex_to_hash(hash_str)
    except Exception:
        return None


def _compute_hash_from_storage(img: dict, algorithm: str):
    """
    Download the image from its storage adapter and compute the perceptual hash.
    Returns an imagehash object or None on failure.
    """
    try:
        import imagehash
        from PIL import Image as PILImage
        from storage.services import get_storage_adapter_for_profile
        from storage.models import StorageProfile

        profile = StorageProfile.objects.get(pk=img["storage_profile_id"])
        adapter = get_storage_adapter_for_profile(profile)
        file_data = adapter.download_file(img["storage_key"])

        pil_img = PILImage.open(BytesIO(file_data)).convert("RGB")

        fn_map = {
            "ahash": imagehash.average_hash,
            "phash": imagehash.phash,
            "dhash": imagehash.dhash,
            "whash": imagehash.whash,
        }
        return fn_map[algorithm](pil_img)

    except Exception as exc:
        logger.warning(
            "Hash computation failed for image %s: %s",
            img.get("image_id"), exc,
        )
        return None


def _greedy_cluster(
    hash_map: list[tuple[str, object]],
    threshold: int,
) -> list[dict]:
    """
    O(n·k) greedy clustering identical to doppix's ``cluster_images`` core.

    Returns a list of dicts::

        {
            "rep_id":     str,         # image_id of representative
            "rep_hash":   ImageHash,
            "members":    [            # includes representative
                {"image_id": str, "score": float},
                ...
            ]
        }
    """
    clusters: list[dict] = []

    for image_id, h in hash_map:
        placed = False
        for cluster in clusters:
            dist = abs(h - cluster["rep_hash"])
            if dist <= threshold:
                score = max(0.0, 1.0 - (dist / 64.0))
                cluster["members"].append({"image_id": image_id, "score": score})
                placed = True
                break

        if not placed:
            clusters.append({
                "rep_id":   image_id,
                "rep_hash": h,
                "members":  [{"image_id": image_id, "score": 1.0}],
            })

    # Sort largest clusters first
    clusters.sort(key=lambda c: len(c["members"]), reverse=True)
    return clusters


def _persist_clusters(scan, raw_clusters: list[dict]):
    """
    Write clusters and their members to the DB in a single transaction.

    Existing clusters from a previous run of the same scan are purged first
    (shouldn't happen normally, but guards against partial retries).
    """
    from similarity.models import SimilarityCluster, SimilarityClusterMember, MemberRole
    from images.models import Image

    # Build a lookup from image_id string → Image PK
    all_image_ids = {
        m["image_id"]
        for cluster in raw_clusters
        for m in cluster["members"]
    }
    image_pk_map: dict[str, int] = {
        str(row["image_id"]): row["id"]
        for row in Image.objects.filter(
            image_id__in=all_image_ids,
            organization=scan.organization,
        ).values("id", "image_id")
    }

    with transaction.atomic():
        # Purge any stale data from a partial previous run
        SimilarityCluster.objects.filter(scan=scan).delete()

        clusters_to_create   = []
        members_to_create    = []

        for raw in raw_clusters:
            member_scores = [m["score"] for m in raw["members"] if m["image_id"] != raw["rep_id"]]
            avg_sim = (sum(member_scores) / len(member_scores)) if member_scores else 1.0
            max_sim = max(member_scores) if member_scores else 1.0

            rep_pk = image_pk_map.get(raw["rep_id"])
            if rep_pk is None:
                continue

            cluster = SimilarityCluster(
                scan=scan,
                representative_id=rep_pk,
                member_count=len(raw["members"]),
                avg_similarity=round(avg_sim, 4),
                max_similarity=round(max_sim, 4),
            )
            clusters_to_create.append((cluster, raw["members"], raw["rep_id"]))

        # Bulk-create cluster rows and fetch back their PKs
        created = SimilarityCluster.objects.bulk_create(
            [c for c, _, _ in clusters_to_create]
        )

        for cluster_obj, members, rep_id in zip(created, [m for _, m, _ in clusters_to_create], [r for _, _, r in clusters_to_create]):
            for m in members:
                img_pk = image_pk_map.get(m["image_id"])
                if img_pk is None:
                    continue
                members_to_create.append(
                    SimilarityClusterMember(
                        cluster=cluster_obj,
                        image_id=img_pk,
                        role=(
                            MemberRole.REPRESENTATIVE
                            if m["image_id"] == rep_id
                            else MemberRole.DUPLICATE
                        ),
                        similarity_score=round(m["score"], 4),
                    )
                )

        SimilarityClusterMember.objects.bulk_create(members_to_create, ignore_conflicts=True)