"""
similarity/models.py
====================
Django ORM models for the perceptual-similarity subsystem.

Design decisions
----------------
* A ``SimilarityScan`` is always scoped to an ``Organization`` and optionally
  narrowed to a ``Project``.  This mirrors how every other resource in the
  platform is org-isolated.

* Hashes are stored on ``ImageHash`` — one row per (image, algorithm).  This
  lets us re-use precomputed hashes across multiple scans without re-hashing
  the image from storage on every run.

* ``SimilarityCluster`` / ``SimilarityClusterMember`` model the N-to-M
  grouping.  A single image can only belong to one cluster per scan
  (enforced by unique_together on ClusterMember).

* ``ClusterAction`` is an append-only audit log of every action a user takes
  on a cluster (archive, delete, mark_reviewed, set_representative).  This
  gives us a full history without relying on field mutations alone.

* Status fields use TextChoices so the DB stores readable strings, not
  magic integers.
"""

import uuid
from django.db import models
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _
from images.models import Image
from organizations.models import Organization

# Optional project scope — import lazily to avoid circular deps at module load
# from projects.models import Project  ← done via string reference below

User = get_user_model()


# ---------------------------------------------------------------------------
# Choices
# ---------------------------------------------------------------------------

class HashAlgorithm(models.TextChoices):
    AHASH = "ahash", _("Average Hash (fast)")
    PHASH = "phash", _("Perceptual Hash (robust)")
    DHASH = "dhash", _("Difference Hash (crop-tolerant)")
    WHASH = "whash", _("Wavelet Hash (accurate)")


class ScanScope(models.TextChoices):
    DATALAKE   = "datalake",   _("Entire Datalake")
    PROJECT    = "project",    _("Single Project")


class ScanStatus(models.TextChoices):
    PENDING    = "pending",    _("Pending")
    RUNNING    = "running",    _("Running")
    COMPLETED  = "completed",  _("Completed")
    FAILED     = "failed",     _("Failed")
    CANCELLED  = "cancelled",  _("Cancelled")


class ClusterStatus(models.TextChoices):
    UNREVIEWED = "unreviewed", _("Unreviewed")
    REVIEWED   = "reviewed",   _("Reviewed")
    ACTIONED   = "actioned",   _("Actioned")


class MemberRole(models.TextChoices):
    REPRESENTATIVE = "representative", _("Representative")
    DUPLICATE      = "duplicate",      _("Duplicate")


class ActionType(models.TextChoices):
    ARCHIVE_DUPLICATES    = "archive_duplicates",    _("Archive Duplicates")
    DELETE_DUPLICATES     = "delete_duplicates",     _("Delete Duplicates")
    MARK_REVIEWED         = "mark_reviewed",         _("Mark Reviewed")
    SET_REPRESENTATIVE    = "set_representative",    _("Set Representative")
    REMOVE_FROM_CLUSTER   = "remove_from_cluster",   _("Remove from Cluster")


# ---------------------------------------------------------------------------
# ImageHash  — cached perceptual hash per image per algorithm
# ---------------------------------------------------------------------------

class ImageHash(models.Model):
    """
    Cached perceptual hash for a single image under a specific algorithm.

    Storing hashes separately from Image avoids re-downloading from storage
    on every scan.  If the image file changes (re-upload with same key),
    the hash must be invalidated and recomputed — see the ``invalidate``
    class-method.
    """

    image = models.ForeignKey(
        Image,
        on_delete=models.CASCADE,
        related_name="perceptual_hashes",
        help_text=_("Image this hash belongs to"),
    )
    algorithm = models.CharField(
        max_length=10,
        choices=HashAlgorithm.choices,
        help_text=_("Hash algorithm used"),
    )
    hash_value = models.CharField(
        max_length=256,
        help_text=_("Hex-encoded hash string"),
    )
    computed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "similarity_image_hash"
        verbose_name = _("Image Hash")
        verbose_name_plural = _("Image Hashes")
        unique_together = [("image", "algorithm")]
        indexes = [
            models.Index(fields=["algorithm", "image"]),
        ]

    def __str__(self):
        return f"{self.image.image_id} [{self.algorithm}] = {self.hash_value[:16]}…"

    @classmethod
    def invalidate(cls, image: Image, algorithm: str | None = None):
        """
        Delete cached hash(es) for *image*.
        Pass *algorithm* to invalidate a single algorithm; omit to clear all.
        """
        qs = cls.objects.filter(image=image)
        if algorithm:
            qs = qs.filter(algorithm=algorithm)
        qs.delete()


# ---------------------------------------------------------------------------
# SimilarityScan  — one scan run
# ---------------------------------------------------------------------------

class SimilarityScan(models.Model):
    """
    Represents a single similarity-scan run within an organization.

    A scan:
    1. Is initiated by a user with at least annotator-level access.
    2. Runs asynchronously (Celery task).
    3. Produces ``SimilarityCluster`` rows when complete.
    4. Is immutable once completed — re-runs create new scan records.
    """

    scan_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True,
        db_index=True,
        help_text=_("Public-facing scan identifier"),
    )

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="similarity_scans",
        help_text=_("Organization that owns this scan"),
    )

    # Optional project scope
    project = models.ForeignKey(
        "projects.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="similarity_scans",
        help_text=_("Project scope (null = entire datalake)"),
    )

    scope = models.CharField(
        max_length=20,
        choices=ScanScope.choices,
        default=ScanScope.DATALAKE,
        help_text=_("Whether scan covers the datalake or a single project"),
    )

    algorithm = models.CharField(
        max_length=10,
        choices=HashAlgorithm.choices,
        default=HashAlgorithm.AHASH,
        help_text=_("Perceptual hash algorithm used"),
    )

    # Threshold stored as integer Hamming distance (0–64).
    # The API accepts a float 0.0–1.0 similarity and converts it:
    #   hamming_threshold = round((1 - similarity) * 64)
    threshold = models.PositiveSmallIntegerField(
        default=6,
        help_text=_("Hamming distance threshold (0=exact, 64=anything)"),
    )

    # Human-readable similarity for display (e.g. 0.80 = 80%)
    similarity_threshold = models.FloatField(
        default=0.80,
        help_text=_("Similarity threshold as fraction 0.0–1.0"),
    )

    status = models.CharField(
        max_length=20,
        choices=ScanStatus.choices,
        default=ScanStatus.PENDING,
        db_index=True,
    )

    # Progress counters (updated by the Celery task)
    total_images    = models.PositiveIntegerField(default=0)
    hashed_images   = models.PositiveIntegerField(default=0)
    clusters_found  = models.PositiveIntegerField(default=0)

    # Celery task ID for cancellation / status polling
    task_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text=_("Celery task ID"),
    )

    error_log = models.TextField(
        blank=True,
        null=True,
        help_text=_("Error detail if scan failed"),
    )

    initiated_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="initiated_scans",
    )

    started_at   = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at   = models.DateTimeField(auto_now_add=True)
    updated_at   = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "similarity_scan"
        verbose_name = _("Similarity Scan")
        verbose_name_plural = _("Similarity Scans")
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["organization", "status"]),
            models.Index(fields=["organization", "project"]),
            models.Index(fields=["scan_id"]),
        ]

    def __str__(self):
        scope = self.project.name if self.project else "datalake"
        return f"Scan {self.scan_id} [{self.organization.slug} / {scope}] ({self.status})"

    # ------------------------------------------------------------------
    # Derived stats
    # ------------------------------------------------------------------

    @property
    def progress_pct(self) -> int:
        """Hashing progress as integer 0–100."""
        if self.total_images == 0:
            return 0
        return min(100, round((self.hashed_images / self.total_images) * 100))

    @property
    def eta_seconds(self) -> int | None:
        """
        Rough ETA based on elapsed time and progress.
        Returns None if not enough data yet.
        """
        if not self.started_at or self.hashed_images == 0:
            return None
        from django.utils import timezone
        elapsed = (timezone.now() - self.started_at).total_seconds()
        rate = self.hashed_images / elapsed          # images/second
        remaining = self.total_images - self.hashed_images
        return round(remaining / rate) if rate > 0 else None

    @property
    def total_duplicates(self) -> int:
        """Total duplicate images across all clusters (members - representatives)."""
        return self.clusters.aggregate(
            total=models.Sum(
                models.F("member_count") - 1,
                output_field=models.IntegerField()
            )
        )["total"] or 0


# ---------------------------------------------------------------------------
# SimilarityCluster  — one group of similar images within a scan
# ---------------------------------------------------------------------------

class SimilarityCluster(models.Model):
    """
    A group of images deemed similar under a completed scan.

    ``member_count`` is denormalised for fast listing queries — updated
    whenever a ``SimilarityClusterMember`` is added or removed.
    """

    cluster_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True,
        db_index=True,
    )

    scan = models.ForeignKey(
        SimilarityScan,
        on_delete=models.CASCADE,
        related_name="clusters",
    )

    representative = models.ForeignKey(
        Image,
        on_delete=models.SET_NULL,
        null=True,
        related_name="representative_in_clusters",
        help_text=_("The image chosen as the keeper for this cluster"),
    )

    # Denormalised for fast ordering / filtering
    member_count = models.PositiveIntegerField(
        default=0,
        db_index=True,
        help_text=_("Total images in this cluster (including representative)"),
    )

    # Average similarity score across all non-representative members
    avg_similarity = models.FloatField(
        null=True,
        blank=True,
        help_text=_("Average pairwise similarity score (0.0–1.0)"),
    )

    # Maximum similarity (closest pair — useful for sorting)
    max_similarity = models.FloatField(
        null=True,
        blank=True,
    )

    status = models.CharField(
        max_length=20,
        choices=ClusterStatus.choices,
        default=ClusterStatus.UNREVIEWED,
        db_index=True,
    )

    reviewed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="reviewed_clusters",
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "similarity_cluster"
        verbose_name = _("Similarity Cluster")
        verbose_name_plural = _("Similarity Clusters")
        ordering = ["-member_count", "-avg_similarity"]
        indexes = [
            models.Index(fields=["scan", "status"]),
            models.Index(fields=["scan", "member_count"]),
        ]

    def __str__(self):
        return f"Cluster {self.cluster_id} [{self.member_count} images]"

    def refresh_member_count(self):
        """Recount members and update the denormalised field."""
        self.member_count = self.members.count()
        self.save(update_fields=["member_count"])


# ---------------------------------------------------------------------------
# SimilarityClusterMember  — image ↔ cluster join table
# ---------------------------------------------------------------------------

class SimilarityClusterMember(models.Model):
    """
    Membership of a single image in a similarity cluster.

    ``similarity_score`` is the Hamming-distance-derived similarity between
    this image and the cluster representative (1.0 for the representative itself).
    """

    cluster = models.ForeignKey(
        SimilarityCluster,
        on_delete=models.CASCADE,
        related_name="members",
    )

    image = models.ForeignKey(
        Image,
        on_delete=models.CASCADE,
        related_name="cluster_memberships",
    )

    role = models.CharField(
        max_length=20,
        choices=MemberRole.choices,
        default=MemberRole.DUPLICATE,
        db_index=True,
    )

    similarity_score = models.FloatField(
        help_text=_("Similarity to cluster representative (1.0 = identical)"),
    )

    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "similarity_cluster_member"
        verbose_name = _("Cluster Member")
        verbose_name_plural = _("Cluster Members")
        unique_together = [("cluster", "image")]
        indexes = [
            models.Index(fields=["cluster", "role"]),
            models.Index(fields=["image"]),
        ]

    def __str__(self):
        return f"{self.image.image_id} in {self.cluster.cluster_id} ({self.role})"


# ---------------------------------------------------------------------------
# ClusterAction  — append-only audit log
# ---------------------------------------------------------------------------

class ClusterAction(models.Model):
    """
    Immutable record of every action taken on a cluster.

    Never updated after creation — new actions always create new rows.
    ``image_ids`` captures the exact set of images affected at the time of
    the action (useful for undo / audit even if images are later deleted).
    """

    action_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True,
    )

    cluster = models.ForeignKey(
        SimilarityCluster,
        on_delete=models.CASCADE,
        related_name="actions",
    )

    action_type = models.CharField(
        max_length=30,
        choices=ActionType.choices,
    )

    performed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name="cluster_actions",
    )

    # Snapshot of affected image IDs at action time
    image_ids = models.JSONField(
        default=list,
        help_text=_("Image IDs affected by this action"),
    )

    # Extra context (e.g. new representative ID for set_representative)
    meta = models.JSONField(
        default=dict,
        blank=True,
    )

    performed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "similarity_cluster_action"
        verbose_name = _("Cluster Action")
        verbose_name_plural = _("Cluster Actions")
        ordering = ["-performed_at"]
        indexes = [
            models.Index(fields=["cluster", "performed_at"]),
            models.Index(fields=["performed_by"]),
        ]

    def __str__(self):
        return f"{self.action_type} on {self.cluster.cluster_id} by {self.performed_by}"