"""
similarity/api/similarity.py
============================
FastAPI routes for the perceptual-similarity feature.

Surfaces
--------
* **Scan management**      POST/GET/DELETE  /similarity/scans
* **Scan progress**        GET              /similarity/scans/{scan_id}
* **Scan results**         GET              /similarity/scans/{scan_id}/results
* **Cluster actions**      POST             /similarity/clusters/{cluster_id}/action
* **Bulk cluster actions** POST             /similarity/clusters/bulk-action

Patterns
--------
All patterns are taken verbatim from ``images.py``:

* ORM calls wrapped with ``@sync_to_async`` — never raw ORM in async context.
* ``ctx.organization`` enforces org-level isolation on every query.
* Responses are hand-serialised dicts — no DRF / Pydantic ORM mode.
* ``ctx.require_role()`` guards destructive operations.
* ``transaction.atomic()`` for multi-row writes.
* Optimistic storage delete AFTER DB commit (delete actions).
"""

import logging
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from asgiref.sync import sync_to_async
from pydantic import BaseModel, Field, field_validator

import django
django.setup()

from api.dependencies import RequestContext, get_request_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/similarity", tags=["Similarity"])


# ============================================================================
# Pydantic request/response schemas
# ============================================================================

class CreateScanRequest(BaseModel):
    scope: str = Field(
        default="datalake",
        description="'datalake' or 'project'",
    )
    project_id: Optional[UUID] = Field(
        default=None,
        description="Required when scope='project'",
    )
    algorithm: str = Field(
        default="ahash",
        description="Hash algorithm: ahash | phash | dhash | whash",
    )
    similarity_threshold: float = Field(
        default=0.80,
        ge=0.50,
        le=1.00,
        description="Similarity threshold 0.50–1.00 (e.g. 0.80 = 80% similar)",
    )

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, v):
        if v not in ("datalake", "project"):
            raise ValueError("scope must be 'datalake' or 'project'")
        return v

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v):
        if v not in ("ahash", "phash", "dhash", "whash"):
            raise ValueError("algorithm must be one of: ahash, phash, dhash, whash")
        return v

    @field_validator("project_id", mode="before")
    @classmethod
    def project_id_required_for_project_scope(cls, v, info):
        # Cross-field validation done in the endpoint after full model parse
        return v


class ClusterActionRequest(BaseModel):
    action: str = Field(
        description=(
            "archive_duplicates | delete_duplicates | "
            "mark_reviewed | set_representative | remove_from_cluster"
        )
    )
    image_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific image IDs to act on (subset of cluster duplicates)",
    )
    new_representative_id: Optional[str] = Field(
        default=None,
        description="Required for set_representative action",
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        allowed = {
            "archive_duplicates", "delete_duplicates",
            "mark_reviewed", "set_representative", "remove_from_cluster",
        }
        if v not in allowed:
            raise ValueError(f"action must be one of: {', '.join(sorted(allowed))}")
        return v


class BulkClusterActionRequest(BaseModel):
    cluster_ids: List[str] = Field(description="List of cluster UUIDs to act on")
    action: str = Field(
        description="archive_duplicates | delete_duplicates | mark_reviewed"
    )

    @field_validator("action")
    @classmethod
    def validate_bulk_action(cls, v):
        allowed = {"archive_duplicates", "delete_duplicates", "mark_reviewed"}
        if v not in allowed:
            raise ValueError(f"Bulk action must be one of: {', '.join(sorted(allowed))}")
        return v


# ============================================================================
# Serialisation helpers
# ============================================================================

def _serialize_scan(scan) -> dict:
    return {
        "scan_id":             str(scan.scan_id),
        "scope":               scan.scope,
        "project_id":          str(scan.project.project_id) if scan.project else None,
        "algorithm":           scan.algorithm,
        "similarity_threshold": scan.similarity_threshold,
        "hamming_threshold":   scan.threshold,
        "status":              scan.status,
        "progress":            scan.progress_pct,
        "total_images":        scan.total_images,
        "hashed_images":       scan.hashed_images,
        "clusters_found":      scan.clusters_found,
        "eta_seconds":         scan.eta_seconds,
        "initiated_by":        scan.initiated_by.username if scan.initiated_by else None,
        "started_at":          scan.started_at.isoformat()  if scan.started_at   else None,
        "completed_at":        scan.completed_at.isoformat() if scan.completed_at else None,
        "created_at":          scan.created_at.isoformat(),
        "error_log":           scan.error_log,
    }


def _serialize_cluster(cluster, include_members: bool = False) -> dict:
    data = {
        "cluster_id":     str(cluster.cluster_id),
        "scan_id":        str(cluster.scan.scan_id),
        "member_count":   cluster.member_count,
        "avg_similarity": round(cluster.avg_similarity or 0, 4),
        "max_similarity": round(cluster.max_similarity or 0, 4),
        "status":         cluster.status,
        "representative": _serialize_image_stub(cluster.representative) if cluster.representative else None,
        "reviewed_by":    cluster.reviewed_by.username if cluster.reviewed_by else None,
        "reviewed_at":    cluster.reviewed_at.isoformat() if cluster.reviewed_at else None,
        "created_at":     cluster.created_at.isoformat(),
    }

    if include_members:
        data["members"] = [
            {
                "image":            _serialize_image_stub(m.image),
                "role":             m.role,
                "similarity_score": round(m.similarity_score, 4),
            }
            for m in cluster.members.select_related(
                "image", "image__storage_profile", "image__uploaded_by"
            ).order_by("-similarity_score")
        ]

    return data


def _serialize_image_stub(image) -> dict:
    """Compact image representation used inside cluster payloads."""
    if image is None:
        return {}
    return {
        "image_id":          str(image.image_id),
        "name":              image.name,
        "original_filename": image.original_filename,
        "file_format":       image.file_format,
        "file_size":         image.file_size,
        "file_size_mb":      round(image.file_size_mb, 2),
        "width":             image.width,
        "height":            image.height,
        "checksum":          image.checksum,
        "storage_key":       image.storage_key,
        "created_at":        image.created_at.isoformat(),
    }


# ============================================================================
# Scan endpoints
# ============================================================================

@router.post(
    "/scans",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Launch Similarity Scan",
    description=(
        "Start an asynchronous perceptual-similarity scan. "
        "The response is returned immediately with status=pending. "
        "Poll GET /similarity/scans/{scan_id} for progress."
    ),
)
async def create_scan(
    payload: CreateScanRequest,
    ctx: RequestContext = Depends(get_request_context),
):
    """
    Launch a similarity scan.

    Requires annotator role or above.
    Only one PENDING/RUNNING scan per organization per scope is allowed
    at a time to prevent resource exhaustion.
    """

    @sync_to_async
    def _create(payload, ctx):
        from similarity.models import SimilarityScan, ScanStatus, ScanScope
        from similarity.services.similarity_scan import create_scan as svc_create_scan

        # Guard: no concurrent scans for same org + scope
        active = SimilarityScan.objects.filter(
            organization=ctx.organization,
            status__in=[ScanStatus.PENDING, ScanStatus.RUNNING],
            scope=payload.scope,
        )
        if payload.project_id:
            active = active.filter(project__project_id=payload.project_id)

        if active.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A scan is already running for this scope. Cancel it before starting a new one.",
            )

        # Resolve project if scoped
        project = None
        if payload.scope == ScanScope.PROJECT:
            if not payload.project_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="project_id is required when scope='project'",
                )
            from projects.models import Project
            try:
                project = Project.objects.get(
                    project_id=payload.project_id,
                    organization=ctx.organization,
                    is_deleted=False,
                )
            except Project.DoesNotExist:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Project {payload.project_id} not found",
                )

        scan = svc_create_scan(
            organization=ctx.organization,
            project=project,
            scope=payload.scope,
            algorithm=payload.algorithm,
            similarity_threshold=payload.similarity_threshold,
            initiated_by=ctx.user,
        )
        return scan

    scan = await _create(payload, ctx)

    # Dispatch Celery task outside the sync_to_async block
    from similarity.tasks import run_similarity_scan
    run_similarity_scan.delay(str(scan.scan_id))

    return {
        "message": "Scan queued successfully",
        **_serialize_scan(scan),
    }


@router.get(
    "/scans",
    summary="List Similarity Scans",
    description="List all similarity scans for the organization, newest first.",
)
async def list_scans(
    ctx: RequestContext = Depends(get_request_context),
    skip:       int            = Query(0,    ge=0),
    limit:      int            = Query(20,   ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    scope:      Optional[str]  = Query(None),
    project_id: Optional[UUID] = Query(None),
):
    @sync_to_async
    def _list(ctx, skip, limit, status_filter, scope, project_id):
        from similarity.models import SimilarityScan

        qs = SimilarityScan.objects.filter(
            organization=ctx.organization
        ).select_related("project", "initiated_by").order_by("-created_at")

        if status_filter:
            qs = qs.filter(status=status_filter)
        if scope:
            qs = qs.filter(scope=scope)
        if project_id:
            qs = qs.filter(project__project_id=project_id)

        total  = qs.count()
        scans  = list(qs[skip:skip + limit])
        return total, scans

    total, scans = await _list(ctx, skip, limit, status_filter, scope, project_id)

    return {
        "total": total,
        "page":      (skip // limit) + 1,
        "page_size": limit,
        "scans":     [_serialize_scan(s) for s in scans],
    }


@router.get(
    "/scans/{scan_id}",
    summary="Get Scan Status",
    description="Get current status and progress of a scan. Poll this endpoint while status='running'.",
)
async def get_scan(
    scan_id: UUID,
    ctx: RequestContext = Depends(get_request_context),
):
    @sync_to_async
    def _get(scan_id, ctx):
        from similarity.models import SimilarityScan
        try:
            return SimilarityScan.objects.select_related(
                "project", "initiated_by"
            ).get(scan_id=scan_id, organization=ctx.organization)
        except SimilarityScan.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scan {scan_id} not found",
            )

    scan = await _get(scan_id, ctx)
    return _serialize_scan(scan)


@router.delete(
    "/scans/{scan_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel Scan",
    description="Cancel a PENDING or RUNNING scan. Completed scans cannot be cancelled.",
)
async def cancel_scan(
    scan_id: UUID,
    ctx: RequestContext = Depends(get_request_context),
):
    @sync_to_async
    def _cancel(scan_id, ctx):
        from similarity.models import SimilarityScan
        from similarity.services.similarity_scan import cancel_scan as svc_cancel

        try:
            scan = SimilarityScan.objects.get(
                scan_id=scan_id, organization=ctx.organization
            )
        except SimilarityScan.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scan {scan_id} not found",
            )

        try:
            return svc_cancel(scan, user=ctx.user)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )

    scan = await _cancel(scan_id, ctx)
    return {"message": "Scan cancelled", **_serialize_scan(scan)}


# ============================================================================
# Scan results  — clusters
# ============================================================================

@router.get(
    "/scans/{scan_id}/results",
    summary="Get Scan Results",
    description=(
        "Retrieve paginated similarity clusters produced by a completed scan. "
        "Supports filtering by status and sorting by size or similarity."
    ),
)
async def get_scan_results(
    scan_id:    UUID,
    ctx:        RequestContext = Depends(get_request_context),
    skip:       int  = Query(0,    ge=0),
    limit:      int  = Query(20,   ge=1, le=100),
    filter_status: Optional[str] = Query(None, alias="status",
                                         description="unreviewed | reviewed | actioned"),
    sort:       str  = Query("size_desc",
                             description="size_desc | size_asc | similarity_desc | date_asc"),
    min_size:   int  = Query(2, ge=2, description="Minimum cluster size"),
):
    @sync_to_async
    def _results(scan_id, ctx, skip, limit, filter_status, sort, min_size):
        from similarity.models import SimilarityScan, SimilarityCluster

        try:
            scan = SimilarityScan.objects.get(
                scan_id=scan_id, organization=ctx.organization
            )
        except SimilarityScan.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scan {scan_id} not found",
            )

        qs = SimilarityCluster.objects.filter(
            scan=scan,
            member_count__gte=min_size,
        ).select_related("scan", "representative", "reviewed_by")

        if filter_status:
            qs = qs.filter(status=filter_status)

        sort_map = {
            "size_desc":       "-member_count",
            "size_asc":        "member_count",
            "similarity_desc": "-avg_similarity",
            "date_asc":        "created_at",
        }
        qs = qs.order_by(sort_map.get(sort, "-member_count"))

        total    = qs.count()
        clusters = list(qs[skip:skip + limit])

        # Summary stats from the scan
        from django.db.models import Sum
        stats = SimilarityCluster.objects.filter(scan=scan).aggregate(
            total_clusters=django.db.models.Count("id"),
            total_members=Sum("member_count"),
        )
        total_duplicates = (stats["total_members"] or 0) - (stats["total_clusters"] or 0)

        return scan, total, clusters, total_duplicates

    import django.db.models
    scan, total, clusters, total_duplicates = await _results(
        scan_id, ctx, skip, limit, filter_status, sort, min_size
    )

    return {
        "scan_id":         str(scan.scan_id),
        "scan_status":     scan.status,
        "total_clusters":  total,
        "total_duplicates": total_duplicates,
        "page":            (skip // limit) + 1,
        "page_size":       limit,
        "clusters":        [_serialize_cluster(c, include_members=False) for c in clusters],
    }


@router.get(
    "/scans/{scan_id}/results/{cluster_id}",
    summary="Get Cluster Detail",
    description="Get full detail for a single cluster, including all member images.",
)
async def get_cluster_detail(
    scan_id:    UUID,
    cluster_id: UUID,
    ctx:        RequestContext = Depends(get_request_context),
):
    @sync_to_async
    def _get_cluster(scan_id, cluster_id, ctx):
        from similarity.models import SimilarityScan, SimilarityCluster

        try:
            scan = SimilarityScan.objects.get(
                scan_id=scan_id, organization=ctx.organization
            )
        except SimilarityScan.DoesNotExist:
            raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")

        try:
            return SimilarityCluster.objects.select_related(
                "scan", "representative", "reviewed_by"
            ).get(cluster_id=cluster_id, scan=scan)
        except SimilarityCluster.DoesNotExist:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    cluster = await _get_cluster(scan_id, cluster_id, ctx)

    @sync_to_async
    def _enrich(cluster):
        # Fetch action history
        actions = list(
            cluster.actions.select_related("performed_by").order_by("-performed_at")[:20]
        )
        return cluster, actions

    cluster, actions = await _enrich(cluster)

    data = _serialize_cluster(cluster, include_members=True)
    data["action_history"] = [
        {
            "action_id":    str(a.action_id),
            "action_type":  a.action_type,
            "performed_by": a.performed_by.username if a.performed_by else None,
            "image_ids":    a.image_ids,
            "meta":         a.meta,
            "performed_at": a.performed_at.isoformat(),
        }
        for a in actions
    ]
    return data


# ============================================================================
# Cluster action endpoints
# ============================================================================

@router.post(
    "/clusters/{cluster_id}/action",
    status_code=status.HTTP_200_OK,
    summary="Perform Cluster Action",
    description=(
        "Apply an action to a single cluster. "
        "Destructive actions (delete_duplicates) require admin or owner role."
    ),
)
async def cluster_action(
    cluster_id: UUID,
    payload:    ClusterActionRequest,
    ctx:        RequestContext = Depends(get_request_context),
):
    # Destructive actions require elevated role
    if payload.action == "delete_duplicates":
        ctx.require_role("admin", "owner")

    @sync_to_async
    def _action(cluster_id, payload, ctx):
        from similarity.models import SimilarityCluster
        from similarity.services.similarity_scan import action_cluster

        try:
            cluster = SimilarityCluster.objects.select_related(
                "scan", "scan__organization", "representative"
            ).get(
                cluster_id=cluster_id,
                scan__organization=ctx.organization,
            )
        except SimilarityCluster.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster {cluster_id} not found",
            )

        try:
            return action_cluster(
                cluster=cluster,
                action_type=payload.action,
                performed_by=ctx.user,
                image_ids=payload.image_ids,
                new_representative_id=payload.new_representative_id,
            ), cluster
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )

    result, cluster = await _action(cluster_id, payload, ctx)

    return {
        "message":    f"Action '{payload.action}' applied successfully",
        "cluster_id": str(cluster_id),
        "action":     payload.action,
        **result,
    }


@router.post(
    "/clusters/bulk-action",
    status_code=status.HTTP_200_OK,
    summary="Bulk Cluster Action",
    description=(
        "Apply the same action to multiple clusters at once. "
        "delete_duplicates requires admin or owner role."
    ),
)
async def bulk_cluster_action(
    payload: BulkClusterActionRequest,
    ctx:     RequestContext = Depends(get_request_context),
):
    if payload.action == "delete_duplicates":
        ctx.require_role("admin", "owner")

    if not payload.cluster_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="cluster_ids must not be empty",
        )

    @sync_to_async
    def _bulk(payload, ctx):
        from similarity.services.similarity_scan import bulk_action_clusters

        try:
            return bulk_action_clusters(
                cluster_ids=payload.cluster_ids,
                action_type=payload.action,
                organization=ctx.organization,
                performed_by=ctx.user,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )

    result = await _bulk(payload, ctx)

    return {
        "message": f"Bulk action '{payload.action}' completed",
        "action":  payload.action,
        **result,
    }


# ============================================================================
# Image-centric similarity endpoints
# (check a single image's similarity against the datalake)
# ============================================================================

@router.get(
    "/images/{image_id}/similar",
    summary="Find Similar Images",
    description=(
        "Find images similar to a specific image using its cached perceptual hash. "
        "Returns results from the most recent completed scan, or computes on-the-fly "
        "if no scan exists."
    ),
)
async def get_similar_images(
    image_id:   UUID,
    ctx:        RequestContext = Depends(get_request_context),
    threshold:  float = Query(0.80, ge=0.50, le=1.00),
    algorithm:  str   = Query("ahash"),
    limit:      int   = Query(20,  ge=1, le=100),
):
    @sync_to_async
    def _find_similar(image_id, ctx, threshold, algorithm, limit):
        from similarity.models import ImageHash, SimilarityClusterMember
        from images.models import Image
        import imagehash

        # Fetch the target image
        try:
            image = Image.objects.get(
                image_id=image_id,
                organization=ctx.organization,
            )
        except Image.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image {image_id} not found",
            )

        # Try cached hash first
        try:
            cached = ImageHash.objects.get(image=image, algorithm=algorithm)
            target_hash = imagehash.hex_to_hash(cached.hash_value)
        except ImageHash.DoesNotExist:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"No cached hash found for image {image_id} with algorithm '{algorithm}'. "
                    "Run a similarity scan first, or upload the image again."
                ),
            )

        hamming_threshold = round((1.0 - threshold) * 64)

        # Fetch all hashes for this org + algorithm (excluding the target)
        all_hashes = ImageHash.objects.filter(
            image__organization=ctx.organization,
            algorithm=algorithm,
        ).exclude(image=image).select_related("image", "image__storage_profile")

        results = []
        for ih in all_hashes:
            try:
                h = imagehash.hex_to_hash(ih.hash_value)
                dist = abs(target_hash - h)
                if dist <= hamming_threshold:
                    score = max(0.0, 1.0 - (dist / 64.0))
                    results.append((score, ih.image))
            except Exception:
                continue

        results.sort(key=lambda x: x[0], reverse=True)
        results = results[:limit]

        return image, results

    target_image, similar = await _find_similar(image_id, ctx, threshold, algorithm, limit)

    return {
        "image_id":   str(image_id),
        "algorithm":  algorithm,
        "threshold":  threshold,
        "total_found": len(similar),
        "similar_images": [
            {
                "similarity_score": round(score, 4),
                **_serialize_image_stub(img),
            }
            for score, img in similar
        ],
    }


# ============================================================================
# Statistics
# ============================================================================

@router.get(
    "/stats",
    summary="Similarity Statistics",
    description="Summary statistics for similarity scans in the organization.",
)
async def get_similarity_stats(
    ctx:        RequestContext = Depends(get_request_context),
    project_id: Optional[UUID] = Query(None),
):
    @sync_to_async
    def _stats(ctx, project_id):
        from similarity.models import SimilarityScan, SimilarityCluster, ScanStatus, ClusterStatus
        from django.db.models import Count, Sum, Avg

        scan_qs = SimilarityScan.objects.filter(organization=ctx.organization)
        if project_id:
            scan_qs = scan_qs.filter(project__project_id=project_id)

        scan_stats = scan_qs.aggregate(
            total_scans=Count("id"),
            completed_scans=Count("id", filter=django.db.models.Q(status=ScanStatus.COMPLETED)),
            running_scans=Count("id", filter=django.db.models.Q(status=ScanStatus.RUNNING)),
        )

        cluster_qs = SimilarityCluster.objects.filter(scan__organization=ctx.organization)
        if project_id:
            cluster_qs = cluster_qs.filter(scan__project__project_id=project_id)

        cluster_stats = cluster_qs.aggregate(
            total_clusters=Count("id"),
            unreviewed=Count("id", filter=django.db.models.Q(status=ClusterStatus.UNREVIEWED)),
            reviewed=Count("id",   filter=django.db.models.Q(status=ClusterStatus.REVIEWED)),
            actioned=Count("id",   filter=django.db.models.Q(status=ClusterStatus.ACTIONED)),
            total_members=Sum("member_count"),
            avg_cluster_size=Avg("member_count"),
        )

        total_clusters = cluster_stats["total_clusters"] or 0
        total_members  = cluster_stats["total_members"]  or 0
        total_duplicates = total_members - total_clusters

        latest_scan = scan_qs.filter(status=ScanStatus.COMPLETED).order_by("-completed_at").first()

        return scan_stats, cluster_stats, total_duplicates, latest_scan

    import django.db.models
    scan_stats, cluster_stats, total_duplicates, latest_scan = await _stats(ctx, project_id)

    return {
        "scans": {
            "total":     scan_stats["total_scans"]     or 0,
            "completed": scan_stats["completed_scans"] or 0,
            "running":   scan_stats["running_scans"]   or 0,
        },
        "clusters": {
            "total":            cluster_stats["total_clusters"]  or 0,
            "unreviewed":       cluster_stats["unreviewed"]       or 0,
            "reviewed":         cluster_stats["reviewed"]         or 0,
            "actioned":         cluster_stats["actioned"]         or 0,
            "avg_cluster_size": round(cluster_stats["avg_cluster_size"] or 0, 1),
        },
        "total_duplicates": total_duplicates,
        "latest_scan": {
            "scan_id":      str(latest_scan.scan_id),
            "completed_at": latest_scan.completed_at.isoformat(),
            "algorithm":    latest_scan.algorithm,
            "threshold":    latest_scan.similarity_threshold,
        } if latest_scan else None,
    }