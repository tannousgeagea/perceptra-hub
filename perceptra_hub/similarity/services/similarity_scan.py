"""
similarity/services.py
======================
Business-logic layer for cluster actions.

All functions are synchronous Django ORM calls — they are wrapped with
``sync_to_async`` at the FastAPI layer, exactly as in ``images.py``.

Keeping action logic here (rather than inline in the router) means:
- The router stays thin and readable.
- Logic is testable without spinning up FastAPI.
- Bulk actions and single-cluster actions share the same code path.
"""

import logging
from django.db import transaction
from django.utils import timezone

from similarity.models import (
    SimilarityScan, SimilarityCluster, SimilarityClusterMember,
    ClusterAction, ClusterStatus, ActionType, MemberRole,
    ScanStatus,
)
from images.models import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scan lifecycle
# ---------------------------------------------------------------------------

def create_scan(
    *,
    organization,
    project,
    scope: str,
    algorithm: str,
    similarity_threshold: float,
    initiated_by,
) -> SimilarityScan:
    """
    Create and persist a ``SimilarityScan``, converting the float similarity
    threshold to an integer Hamming distance for the clustering algorithm.

    Enforces at most one PENDING/RUNNING scan per org+scope using
    ``select_for_update`` to prevent race conditions under concurrent requests.

    Does NOT dispatch the Celery task — the caller is responsible for that
    so the scan_id is always available in the response before the task starts.
    """
    hamming = _similarity_to_hamming(similarity_threshold)

    with transaction.atomic():
        # Lock any existing active rows so concurrent requests block here
        # rather than racing past the exists() check.
        active_qs = SimilarityScan.objects.select_for_update().filter(
            organization=organization,
            status__in=[ScanStatus.PENDING, ScanStatus.RUNNING],
            scope=scope,
        )
        if project:
            active_qs = active_qs.filter(project=project)

        if active_qs.exists():
            raise ValueError(
                "A scan is already running for this scope. Cancel it before starting a new one."
            )

        scan = SimilarityScan.objects.create(
            organization=organization,
            project=project,
            scope=scope,
            algorithm=algorithm,
            similarity_threshold=similarity_threshold,
            threshold=hamming,
            initiated_by=initiated_by,
            status=ScanStatus.PENDING,
        )

    logger.info("Created scan %s (org=%s, algo=%s, threshold=%.2f → hamming=%d)",
                scan.scan_id, organization.slug, algorithm, similarity_threshold, hamming)
    return scan


def recluster(
    scan: SimilarityScan,
    new_similarity_threshold: float,
) -> SimilarityScan:
    """
    Queue a recluster run on a completed scan with a new similarity threshold.

    Reuses existing ``ImageHash`` rows — no images are re-downloaded.
    The scan's threshold fields are updated and its status is reset to PENDING.
    """
    if scan.status not in (ScanStatus.COMPLETED, ScanStatus.FAILED):
        raise ValueError(
            f"Only completed or failed scans can be reclustered (current status: '{scan.status}')"
        )

    hamming = _similarity_to_hamming(new_similarity_threshold)
    scan.threshold             = hamming
    scan.similarity_threshold  = new_similarity_threshold
    scan.status                = ScanStatus.PENDING
    scan.clusters_found        = 0
    scan.completed_at          = None
    scan.error_log             = None
    scan.save(update_fields=[
        "threshold", "similarity_threshold", "status",
        "clusters_found", "completed_at", "error_log", "updated_at",
    ])
    logger.info(
        "Queued recluster for scan %s (new threshold=%.2f → hamming=%d)",
        scan.scan_id, new_similarity_threshold, hamming,
    )
    return scan


def cancel_scan(scan: SimilarityScan, user) -> SimilarityScan:
    """
    Cancel a PENDING or RUNNING scan.

    For RUNNING scans the Celery task checks the status field at each
    progress batch and exits gracefully when it sees CANCELLED.
    """
    if scan.status not in (ScanStatus.PENDING, ScanStatus.RUNNING):
        raise ValueError(f"Cannot cancel scan in status '{scan.status}'")

    scan.status = ScanStatus.CANCELLED
    scan.save(update_fields=["status", "updated_at"])

    if scan.task_id:
        try:
            from api.config.celery_utils import celery_app
            celery_app.control.revoke(scan.task_id, terminate=True, signal="SIGTERM")
        except Exception as exc:
            logger.warning("Could not revoke Celery task %s: %s", scan.task_id, exc)

    return scan


# ---------------------------------------------------------------------------
# Cluster actions
# ---------------------------------------------------------------------------

def action_cluster(
    *,
    cluster: SimilarityCluster,
    action_type: str,
    performed_by,
    image_ids: list[str] | None = None,
    new_representative_id: str | None = None,
) -> dict:
    """
    Execute an action on a single cluster and return a summary dict.

    Dispatches to the appropriate handler based on ``action_type``.
    """
    handlers = {
        ActionType.ARCHIVE_DUPLICATES:  _archive_duplicates,
        ActionType.DELETE_DUPLICATES:   _delete_duplicates,
        ActionType.MARK_REVIEWED:       _mark_reviewed,
        ActionType.SET_REPRESENTATIVE:  _set_representative,
        ActionType.REMOVE_FROM_CLUSTER: _remove_from_cluster,
    }

    handler = handlers.get(action_type)
    if not handler:
        raise ValueError(f"Unknown action type: {action_type}")

    return handler(
        cluster=cluster,
        performed_by=performed_by,
        image_ids=image_ids,
        new_representative_id=new_representative_id,
    )


def bulk_action_clusters(
    *,
    cluster_ids: list[str],
    action_type: str,
    organization,
    performed_by,
    atomic: bool = False,
) -> dict:
    """
    Apply ``action_type`` to multiple clusters identified by their UUIDs.

    Parameters
    ----------
    atomic:
        If True, wraps all cluster actions in a single ``transaction.atomic``
        block — either all succeed or none do.  Default is False (best-effort,
        partial success reported).

    Returns aggregated counts.
    """
    clusters = list(
        SimilarityCluster.objects.filter(
            cluster_id__in=cluster_ids,
            scan__organization=organization,
        ).select_related("scan", "representative")
    )

    if not clusters:
        raise ValueError("No matching clusters found")

    results = {"total": len(clusters), "successful": 0, "failed": 0, "details": []}

    def _run_all():
        for cluster in clusters:
            try:
                summary = action_cluster(
                    cluster=cluster,
                    action_type=action_type,
                    performed_by=performed_by,
                )
                results["successful"] += 1
                results["details"].append({"cluster_id": str(cluster.cluster_id), "status": "ok", **summary})
            except Exception as exc:
                results["failed"] += 1
                results["details"].append({
                    "cluster_id": str(cluster.cluster_id),
                    "status": "error",
                    "error": str(exc),
                })
                logger.error("Bulk action %s failed on cluster %s: %s", action_type, cluster.cluster_id, exc)
                if atomic:
                    raise  # bubble up to abort the transaction

    if atomic:
        with transaction.atomic():
            _run_all()
    else:
        _run_all()

    return results


# ---------------------------------------------------------------------------
# Individual action handlers (private)
# ---------------------------------------------------------------------------

def _archive_duplicates(*, cluster, performed_by, image_ids=None, **_) -> dict:
    """
    Add an 'archived' tag to all duplicate images in the cluster and mark
    the cluster as actioned.

    Archiving is non-destructive — images remain in the DB and in storage.
    """
    from images.models import Tag, ImageTag

    duplicate_members = _resolve_duplicate_members(cluster, image_ids)
    duplicate_images  = [m.image for m in duplicate_members]
    affected_ids      = [str(img.image_id) for img in duplicate_images]

    with transaction.atomic():
        # Ensure "archived" tag exists for the org
        org = cluster.scan.organization
        tag, _ = Tag.objects.get_or_create(
            organization=org,
            name="archived",
            defaults={"color": "#6B7280"},
        )

        for img in duplicate_images:
            ImageTag.objects.get_or_create(
                image=img,
                tag=tag,
                defaults={"tagged_by": performed_by},
            )

        _mark_cluster_actioned(cluster, performed_by)
        _log_action(cluster, ActionType.ARCHIVE_DUPLICATES, performed_by, affected_ids)

    logger.info("Archived %d duplicates in cluster %s", len(duplicate_images), cluster.cluster_id)
    return {"archived": len(duplicate_images), "affected_image_ids": affected_ids}


def _delete_duplicates(*, cluster, performed_by, image_ids=None, **_) -> dict:
    """
    Permanently delete duplicate images (storage + DB rows).

    Deletion order: storage first, then DB (mirrors ``images.py`` bulk delete).
    The representative is always protected.
    """
    duplicate_members = _resolve_duplicate_members(cluster, image_ids)
    duplicate_images  = [m.image for m in duplicate_members]
    affected_ids      = [str(img.image_id) for img in duplicate_images]

    storage_deleted = 0
    storage_failed  = 0

    # Collect storage info before any DB delete
    storage_items = [(img.storage_key, img) for img in duplicate_images if img.storage_key]

    with transaction.atomic():
        _mark_cluster_actioned(cluster, performed_by)
        _log_action(cluster, ActionType.DELETE_DUPLICATES, performed_by, affected_ids)

        # Remove cluster members first (FK CASCADE would handle this, but explicit is cleaner)
        SimilarityClusterMember.objects.filter(
            cluster=cluster,
            image__in=duplicate_images,
        ).delete()

        # Delete Image DB rows
        Image.objects.filter(image_id__in=[img.image_id for img in duplicate_images]).delete()

    # After commit → delete from storage
    for storage_key, img in storage_items:
        try:
            img.delete_from_storage()
            storage_deleted += 1
        except Exception as exc:
            logger.error("Storage delete failed for %s: %s", storage_key, exc)
            storage_failed += 1

    cluster.refresh_member_count()

    logger.info(
        "Deleted %d duplicates from cluster %s (storage_deleted=%d, storage_failed=%d)",
        len(duplicate_images), cluster.cluster_id, storage_deleted, storage_failed,
    )
    return {
        "deleted_from_db": len(duplicate_images),
        "storage_deleted": storage_deleted,
        "storage_failed": storage_failed,
        "affected_image_ids": affected_ids,
    }


def _mark_reviewed(*, cluster, performed_by, **_) -> dict:
    """Mark a cluster as reviewed without taking any destructive action."""
    with transaction.atomic():
        cluster.status      = ClusterStatus.REVIEWED
        cluster.reviewed_by = performed_by
        cluster.reviewed_at = timezone.now()
        cluster.save(update_fields=["status", "reviewed_by", "reviewed_at"])
        _log_action(cluster, ActionType.MARK_REVIEWED, performed_by, [])

    return {"status": ClusterStatus.REVIEWED}


def _set_representative(*, cluster, performed_by, new_representative_id=None, **_) -> dict:
    """
    Swap the representative of a cluster.

    The old representative becomes a DUPLICATE member; the new one becomes
    REPRESENTATIVE.  Cluster.representative FK is updated accordingly.
    """
    if not new_representative_id:
        raise ValueError("new_representative_id is required for set_representative action")

    try:
        new_member = SimilarityClusterMember.objects.select_related("image").get(
            cluster=cluster,
            image__image_id=new_representative_id,
        )
    except SimilarityClusterMember.DoesNotExist:
        raise ValueError(f"Image {new_representative_id} is not a member of this cluster")

    with transaction.atomic():
        # Demote current representative
        SimilarityClusterMember.objects.filter(
            cluster=cluster,
            role=MemberRole.REPRESENTATIVE,
        ).update(role=MemberRole.DUPLICATE)

        # Promote new representative
        new_member.role = MemberRole.REPRESENTATIVE
        new_member.save(update_fields=["role"])

        cluster.representative = new_member.image
        cluster.save(update_fields=["representative"])

        _log_action(
            cluster, ActionType.SET_REPRESENTATIVE, performed_by,
            [new_representative_id],
            meta={"new_representative_id": new_representative_id},
        )

    return {"new_representative_id": new_representative_id}


def _remove_from_cluster(*, cluster, performed_by, image_ids=None, **_) -> dict:
    """
    Remove specific images from a cluster without deleting them.

    If the representative is removed, the member with the highest similarity
    score is promoted automatically.
    """
    if not image_ids:
        raise ValueError("image_ids is required for remove_from_cluster action")

    members_to_remove = list(
        SimilarityClusterMember.objects.filter(
            cluster=cluster,
            image__image_id__in=image_ids,
        ).select_related("image")
    )

    affected_ids = [str(m.image.image_id) for m in members_to_remove]
    rep_removed  = any(m.role == MemberRole.REPRESENTATIVE for m in members_to_remove)

    with transaction.atomic():
        SimilarityClusterMember.objects.filter(
            cluster=cluster,
            image__image_id__in=image_ids,
        ).delete()

        # Auto-promote new representative if needed
        if rep_removed:
            next_member = (
                SimilarityClusterMember.objects.filter(cluster=cluster)
                .order_by("-similarity_score")
                .select_related("image")
                .first()
            )
            if next_member:
                next_member.role = MemberRole.REPRESENTATIVE
                next_member.save(update_fields=["role"])
                cluster.representative = next_member.image
                cluster.save(update_fields=["representative"])

        cluster.refresh_member_count()
        _log_action(cluster, ActionType.REMOVE_FROM_CLUSTER, performed_by, affected_ids)

    return {"removed": len(members_to_remove), "affected_image_ids": affected_ids}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _resolve_duplicate_members(
    cluster: SimilarityCluster,
    image_ids: list[str] | None,
) -> list[SimilarityClusterMember]:
    """
    Return duplicate ``SimilarityClusterMember`` rows for a cluster.

    If *image_ids* is provided, restrict to those images (still excluding the
    representative).  The representative is ALWAYS protected.
    """
    qs = SimilarityClusterMember.objects.filter(
        cluster=cluster,
        role=MemberRole.DUPLICATE,
    ).select_related("image", "image__storage_profile")

    if image_ids:
        qs = qs.filter(image__image_id__in=image_ids)

    return list(qs)


def _mark_cluster_actioned(cluster: SimilarityCluster, performed_by):
    cluster.status      = ClusterStatus.ACTIONED
    cluster.reviewed_by = performed_by
    cluster.reviewed_at = timezone.now()
    cluster.save(update_fields=["status", "reviewed_by", "reviewed_at"])


def _log_action(
    cluster: SimilarityCluster,
    action_type: str,
    performed_by,
    image_ids: list[str],
    meta: dict | None = None,
):
    ClusterAction.objects.create(
        cluster=cluster,
        action_type=action_type,
        performed_by=performed_by,
        image_ids=image_ids,
        meta=meta or {},
    )


def _similarity_to_hamming(similarity: float) -> int:
    """
    Convert a 0.0–1.0 similarity fraction to a Hamming distance threshold.

    similarity=1.0  → hamming=0  (exact match only)
    similarity=0.80 → hamming=12 (within 12 bits of 64)
    similarity=0.0  → hamming=64 (anything matches)
    """
    similarity = max(0.0, min(1.0, similarity))
    return round((1.0 - similarity) * 64)