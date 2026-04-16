"""
similarity/tasks/similarity_scan.py
====================================
Celery tasks for running perceptual-similarity scans asynchronously.

Architecture
------------
``run_similarity_scan`` is the primary task.  It:

1. Resolves the image queryset from the scan's scope (datalake or project).
2. Hashes each image using imagehash — downloading from storage only when
   no cached ``ImageHash`` row exists.  Downloads run in parallel via a
   ``ThreadPoolExecutor`` (HASH_WORKERS concurrent threads).
3. Runs BK-tree clustering — O(n log k) instead of the O(n·k) greedy loop.
4. Persists ``SimilarityCluster`` + ``SimilarityClusterMember`` rows inside a
   single transaction.
5. Updates ``SimilarityScan.status`` / progress counters throughout.

Progress updates are written directly to the DB every ``PROGRESS_BATCH``
images so the polling endpoint always returns fresh data without Redis pub/sub.

BK-tree clustering
------------------
A BK-tree (Burkhard-Keller tree) is a metric-space index that works with any
distance satisfying the triangle inequality.  Hamming distance does.  For each
new image we descend the tree with ``find(hash, threshold)`` — average cost is
O(log k) where k is the number of existing clusters.  Total scan cost drops
from O(n²) to O(n log n).

Error handling
--------------
* Individual hash failures are counted and reported; they do not abort the scan.
* ``SoftTimeLimitExceeded`` triggers an automatic retry (up to 2 times, 60 s
  between attempts) so a worker restart during a long scan doesn't permanently
  fail it.
* Any other unhandled exception marks the scan as FAILED and stores the
  traceback in ``error_log``.
"""

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Optional

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.conf import settings as django_settings
from django.db import transaction
from django.utils import timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (override via Django settings if needed)
# ---------------------------------------------------------------------------

# How often to flush progress to the DB (every N images processed)
PROGRESS_BATCH: int = getattr(django_settings, "SIMILARITY_PROGRESS_BATCH", 50)

# Number of concurrent storage-download threads during hash computation
HASH_WORKERS: int = getattr(django_settings, "SIMILARITY_HASH_WORKERS", 20)


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@shared_task(
    bind=True,
    name="similarity:run_scan",
    max_retries=2,
    default_retry_delay=60,          # 60 s between retries
    autoretry_for=(SoftTimeLimitExceeded,),
    retry_backoff=False,
    acks_late=True,                  # only ack after task body finishes
    time_limit=3600,                 # hard kill after 1 hour
    soft_time_limit=3540,            # SoftTimeLimitExceeded raised at 59 min
)
def run_similarity_scan(self, scan_id: str) -> dict:
    """
    Execute a similarity scan identified by *scan_id* (UUID string).

    Called by ``POST /similarity/scans`` after the ``SimilarityScan`` row
    is created with status=PENDING.

    Returns a summary dict that Celery stores as the task result.
    """
    from similarity.models import (
        SimilarityScan, ScanStatus, ImageHash,
        SimilarityCluster, SimilarityClusterMember, MemberRole,
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
            _mark_completed(scan, clusters_found=0, failed_hashes=0)
            return {"status": "completed", "clusters": 0, "images": 0, "failed_hashes": 0}

        # ── 3. Load or compute hashes ────────────────────────────────────────
        algorithm = scan.algorithm

        # Fetch all cached hashes for this org + algorithm in one query
        cached: dict[str, str] = {
            str(row["image_id"]): row["hash_value"]
            for row in ImageHash.objects.filter(
                image__organization=scan.organization,
                algorithm=algorithm,
            ).values("image_id", "hash_value")
        }

        hash_map: list[tuple[str, object]] = []   # [(image_id_str, imagehash_obj)]
        to_cache:  list[ImageHash]         = []
        hashed        = 0
        failed_hashes = 0

        # Split into cached (cheap) and uncached (needs download) buckets
        cached_images   = [img for img in images if str(img["image_id"]) in cached]
        uncached_images = [img for img in images if str(img["image_id"]) not in cached]

        # --- 3a. Re-hydrate cached hashes (no I/O) --------------------------
        for img in cached_images:
            img_uuid = str(img["image_id"])
            h = _parse_hash(cached[img_uuid])
            if h is not None:
                hash_map.append((img_uuid, h))
            else:
                failed_hashes += 1
            hashed += 1

        # --- 3b. Download + hash uncached images (parallel) -----------------
        def _process_uncached(img):
            """Returns (img_uuid, img_pk, hash_obj) or (img_uuid, img_pk, None) on failure."""
            raw = _compute_hash_from_storage(img, algorithm)
            return str(img["image_id"]), img["id"], raw

        with ThreadPoolExecutor(max_workers=HASH_WORKERS) as pool:
            futures = {pool.submit(_process_uncached, img): img for img in uncached_images}

            for future in as_completed(futures):
                img_uuid, img_pk, raw_hash = future.result()

                if raw_hash is None:
                    logger.warning("Could not hash image %s — skipping", img_uuid)
                    failed_hashes += 1
                else:
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

                # Flush progress and check for cancellation every PROGRESS_BATCH
                if hashed % PROGRESS_BATCH == 0:
                    scan.refresh_from_db(fields=["status"])
                    if scan.status == ScanStatus.CANCELLED:
                        logger.info("Scan %s cancelled mid-run", scan_id)
                        pool.shutdown(wait=False, cancel_futures=True)
                        return {"status": "cancelled"}

                    scan.hashed_images = hashed
                    scan.save(update_fields=["hashed_images", "updated_at"])

        # Bulk-save newly computed hashes
        if to_cache:
            ImageHash.objects.bulk_create(to_cache, ignore_conflicts=True)

        scan.hashed_images = hashed
        scan.save(update_fields=["hashed_images", "updated_at"])

        # ── 4. BK-tree clustering (pure Python — no DB I/O) ─────────────────
        raw_clusters = _bktree_cluster(hash_map, threshold=scan.threshold)

        # Only keep clusters with ≥2 members
        raw_clusters = [c for c in raw_clusters if len(c["members"]) >= 2]

        # ── 5. Persist clusters ──────────────────────────────────────────────
        _persist_clusters(scan, raw_clusters)

        # ── 6. Mark completed ────────────────────────────────────────────────
        _mark_completed(scan, clusters_found=len(raw_clusters), failed_hashes=failed_hashes)

        logger.info(
            "Scan %s completed: %d images, %d clusters, %d hash failures",
            scan_id, len(images), len(raw_clusters), failed_hashes,
        )
        return {
            "status":        "completed",
            "images":        len(images),
            "clusters":      len(raw_clusters),
            "failed_hashes": failed_hashes,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Scan %s failed: %s", scan_id, exc)
        scan.status    = ScanStatus.FAILED
        scan.error_log = tb
        scan.save(update_fields=["status", "error_log", "updated_at"])
        raise  # re-raise so Celery marks task as FAILURE (or retries on SoftTimeLimitExceeded)


# ---------------------------------------------------------------------------
# Recluster task — reuse existing hashes, skip download phase
# ---------------------------------------------------------------------------

@shared_task(
    bind=True,
    name="similarity:recluster",
    max_retries=2,
    default_retry_delay=60,
    autoretry_for=(SoftTimeLimitExceeded,),
    acks_late=True,
    time_limit=1800,
    soft_time_limit=1740,
)
def recluster_scan(self, scan_id: str, new_threshold: int) -> dict:
    """
    Re-run clustering on an existing completed scan using a new Hamming
    threshold, without re-downloading or re-hashing any images.

    Called by ``POST /similarity/scans/{scan_id}/recluster``.
    """
    from similarity.models import SimilarityScan, ScanStatus, ImageHash

    try:
        scan = SimilarityScan.objects.select_related("organization", "project").get(
            scan_id=scan_id
        )
    except SimilarityScan.DoesNotExist:
        logger.error("Recluster: scan %s not found", scan_id)
        return {"error": "scan_not_found"}

    scan.status   = ScanStatus.RUNNING
    scan.threshold = new_threshold
    scan.similarity_threshold = round(1.0 - (new_threshold / 64.0), 4)
    scan.task_id  = self.request.id
    scan.clusters_found = 0
    scan.started_at = timezone.now()
    scan.completed_at = None
    scan.save(update_fields=[
        "status", "threshold", "similarity_threshold",
        "task_id", "clusters_found", "started_at", "completed_at", "updated_at",
    ])

    try:
        image_qs = _resolve_image_qs(scan)
        images = list(image_qs.values("id", "image_id"))

        cached = {
            str(row["image_id"]): row["hash_value"]
            for row in ImageHash.objects.filter(
                image__organization=scan.organization,
                algorithm=scan.algorithm,
                image__in=[img["id"] for img in images],
            ).values("image_id", "hash_value")
        }

        hash_map = []
        for img in images:
            img_uuid = str(img["image_id"])
            h = _parse_hash(cached.get(img_uuid))
            if h is not None:
                hash_map.append((img_uuid, h))

        raw_clusters = _bktree_cluster(hash_map, threshold=new_threshold)
        raw_clusters = [c for c in raw_clusters if len(c["members"]) >= 2]

        _persist_clusters(scan, raw_clusters)
        _mark_completed(scan, clusters_found=len(raw_clusters), failed_hashes=0)

        logger.info("Recluster %s completed: %d clusters (threshold=%d)", scan_id, len(raw_clusters), new_threshold)
        return {"status": "completed", "clusters": len(raw_clusters)}

    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Recluster %s failed: %s", scan_id, exc)
        scan.status    = ScanStatus.FAILED
        scan.error_log = tb
        scan.save(update_fields=["status", "error_log", "updated_at"])
        raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mark_completed(scan, clusters_found: int, failed_hashes: int = 0):
    from similarity.models import ScanStatus
    scan.status         = ScanStatus.COMPLETED
    scan.clusters_found = clusters_found
    scan.completed_at   = timezone.now()
    # Store failed_hashes in error_log only if there were failures, non-destructively
    if failed_hashes > 0 and not scan.error_log:
        scan.error_log = f"Warning: {failed_hashes} image(s) could not be hashed and were skipped."
    scan.save(update_fields=["status", "clusters_found", "completed_at", "error_log", "updated_at"])


def _parse_hash(hash_str: Optional[str]):
    """Re-hydrate an imagehash object from its hex string."""
    if not hash_str:
        return None
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


def _resolve_image_qs(scan):
    """Return the Image queryset for a scan's scope."""
    from images.models import Image
    qs = Image.objects.filter(organization=scan.organization)
    if scan.project_id:
        qs = qs.filter(
            project_assignments__project=scan.project,
            project_assignments__is_active=True,
        )
    return qs


# ---------------------------------------------------------------------------
# BK-tree  (Burkhard-Keller tree for Hamming distance)
# ---------------------------------------------------------------------------

class _BKNode:
    """Single node in a BK-tree storing an imagehash and its image_id."""
    __slots__ = ("image_id", "h", "children")

    def __init__(self, image_id: str, h):
        self.image_id = image_id
        self.h        = h
        self.children: dict[int, "_BKNode"] = {}


class _BKTree:
    """
    BK-tree indexed by Hamming distance between imagehash objects.

    Insertion: O(depth) ≈ O(log n) amortised.
    Range query (find all within distance d): O(n^(d/D)) where D is the
    hash bit-width — dramatically sub-linear for small d.
    """

    def __init__(self):
        self.root: Optional[_BKNode] = None

    def _dist(self, a, b) -> int:
        return abs(a - b)

    def insert(self, image_id: str, h) -> None:
        node = _BKNode(image_id, h)
        if self.root is None:
            self.root = node
            return
        cur = self.root
        while True:
            d = self._dist(h, cur.h)
            if d == 0:
                # Exact duplicate of the representative — don't add a new branch
                return
            child = cur.children.get(d)
            if child is None:
                cur.children[d] = node
                return
            cur = child

    def find(self, h, threshold: int) -> list[tuple[int, str]]:
        """Return [(distance, image_id)] for all nodes within *threshold*."""
        if self.root is None:
            return []
        results = []
        stack   = [self.root]
        while stack:
            node = stack.pop()
            d = self._dist(h, node.h)
            if d <= threshold:
                results.append((d, node.image_id))
            lo = max(0, d - threshold)
            hi = d + threshold
            for child_d, child in node.children.items():
                if lo <= child_d <= hi:
                    stack.append(child)
        return results


def _bktree_cluster(
    hash_map: list[tuple[str, object]],
    threshold: int,
) -> list[dict]:
    """
    O(n log k) clustering using a BK-tree.

    Each image is inserted into a BK-tree keyed by the cluster representative's
    hash.  On insertion we query for any existing representative within
    *threshold* Hamming distance.  If found, we join the nearest cluster;
    otherwise we start a new cluster with this image as representative.

    Returns a list of dicts::

        {
            "rep_id":   str,          # image_id of representative
            "rep_hash": imagehash,
            "members":  [             # includes representative
                {"image_id": str, "score": float},
                ...
            ]
        }
    """
    tree: _BKTree              = _BKTree()
    cluster_map: dict[str, dict] = {}   # rep_id → cluster dict

    for image_id, h in hash_map:
        matches = tree.find(h, threshold)

        if matches:
            # Join nearest cluster (smallest distance wins)
            matches.sort(key=lambda x: x[0])
            best_dist, rep_id = matches[0]
            score = max(0.0, 1.0 - (best_dist / 64.0))
            cluster_map[rep_id]["members"].append({"image_id": image_id, "score": score})
        else:
            # New cluster — this image is the representative
            tree.insert(image_id, h)
            cluster_map[image_id] = {
                "rep_id":   image_id,
                "rep_hash": h,
                "members":  [{"image_id": image_id, "score": 1.0}],
            }

    clusters = list(cluster_map.values())
    clusters.sort(key=lambda c: len(c["members"]), reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _persist_clusters(scan, raw_clusters: list[dict]):
    """
    Write clusters and their members to the DB in a single transaction.

    Existing clusters from a previous run of the same scan are purged first
    (guards against partial retries / recluster runs).
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
        # Purge any stale data from a previous run / recluster
        SimilarityCluster.objects.filter(scan=scan).delete()

        clusters_to_create = []
        members_to_create  = []

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

        # Bulk-create cluster rows and get back their PKs
        created = SimilarityCluster.objects.bulk_create(
            [c for c, _, _ in clusters_to_create]
        )

        for cluster_obj, members, rep_id in zip(
            created,
            [m for _, m, _ in clusters_to_create],
            [r for _, _, r in clusters_to_create],
        ):
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
