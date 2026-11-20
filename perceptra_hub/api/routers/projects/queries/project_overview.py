"""
One project's pipeline at a glance, for the console's Overview page.

    GET /api/v1/projects/{project_id}/overview

Counts what already exists at each stage of the loop: data -> annotation ->
review -> dataset versions -> training -> evaluation -> deployment ->
monitoring. Read-only; anyone who can open the project may call it. The
console decides which next actions to offer (it knows the caller's
permissions); this only reports the state.
"""
from datetime import timedelta
from typing import Any, Dict, Optional
from uuid import UUID

from asgiref.sync import sync_to_async
from django.db.models import Count, Q
from django.utils import timezone
from fastapi import APIRouter, Depends

from api.dependencies import ProjectContext, get_project_context

router = APIRouter(prefix="/projects")

# Which metric compares versions: the first of these a version has.
METRIC_PREFERENCE = ("mAP50", "map50", "mAP50-95", "map", "f1", "f1_score", "precision", "accuracy")
REVIEWED_STATUSES = ("reviewed", "approved", "dataset")
RUNNING_TRAINING = ("pending", "queued", "initializing", "running")


def _metric(metrics: Any, name: Optional[str] = None):
    """(name, value) of the comparison metric in a version's metrics, or (None, None)."""
    if not isinstance(metrics, dict):
        return None, None
    for key in ((name,) if name else METRIC_PREFERENCE):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    return None, None


def _version_ref(version) -> Dict[str, Any]:
    return {
        "model_id": version.model.model_id,
        "model_name": version.model.name,
        "version_id": version.version_id,
        "version_number": version.version_number,
        "version_name": version.version_name or f"v{version.version_number}",
    }


def _data_and_annotation(project) -> Dict[str, Any]:
    from jobs.models import Job
    from projects.models import ProjectImage, VersionImage

    week_ago = timezone.now() - timedelta(days=7)
    images = ProjectImage.objects.filter(project=project, is_active=True)
    counts = images.aggregate(
        total=Count("id"),
        added_7d=Count("id", filter=Q(added_at__gte=week_ago)),
        empty=Count("id", filter=Q(marked_as_null=True)),
        unannotated=Count("id", filter=Q(status="unannotated", marked_as_null=False)),
        unannotated_no_job=Count("id", filter=Q(status="unannotated", marked_as_null=False,
                                                job_assignment_status="waiting")),
        in_progress=Count("id", filter=Q(status="in_progress")),
        to_review=Count("id", filter=Q(status="annotated")),
        rejected=Count("id", filter=Q(status="rejected")),
        reviewed=Count("id", filter=Q(status__in=REVIEWED_STATUSES)),
    )
    last_added = images.order_by("-added_at").values_list("added_at", flat=True).first()

    jobs = Job.objects.filter(project=project, is_active=True)
    job_counts = dict(jobs.values_list("status").annotate(n=Count("id")).values_list("status", "n"))
    open_jobs = sum(n for status, n in job_counts.items() if status in ("unassigned", "assigned", "in_review"))

    in_a_version = VersionImage.objects.filter(version__project=project).values("project_image")
    reviewed_not_in_version = images.filter(status__in=REVIEWED_STATUSES).exclude(id__in=in_a_version).count()

    return {
        "data": {
            "total_images": counts["total"],
            "added_7d": counts["added_7d"],
            "last_added_at": last_added,
        },
        "annotation": {
            "unannotated": counts["unannotated"],
            "unannotated_without_job": counts["unannotated_no_job"],
            "in_progress": counts["in_progress"],
            "marked_empty": counts["empty"],
            "open_jobs": open_jobs,
            "jobs_by_status": job_counts,
        },
        "review": {
            "to_review": counts["to_review"],
            "rejected": counts["rejected"],
            "reviewed": counts["reviewed"],
        },
        "_reviewed_not_in_version": reviewed_not_in_version,
    }


def _dataset(project, reviewed_not_in_version: int) -> Dict[str, Any]:
    from projects.models import Version

    versions = Version.objects.filter(project=project)
    latest = versions.order_by("-version_number").first()
    return {
        "versions": versions.count(),
        "latest": None if latest is None else {
            "version_id": str(latest.version_id),
            "version_name": latest.version_name,
            "version_number": latest.version_number,
            "total_images": latest.total_images,
            "export_status": latest.export_status,
            "created_at": latest.created_at,
        },
        "reviewed_not_in_version": reviewed_not_in_version,
    }


def _training(project) -> Dict[str, Any]:
    from training.models import TrainingSession

    sessions = TrainingSession.objects.filter(
        model_version__model__project=project, model_version__model__is_deleted=False
    ).select_related("model_version__model")
    latest = sessions.order_by("-created_at").first()
    out = {"running": sessions.filter(status__in=RUNNING_TRAINING).count(), "latest": None}
    if latest is not None:
        eta = latest.estimated_time_remaining
        out["latest"] = {
            **_version_ref(latest.model_version),
            # The session page and /training-sessions/{id} use this numeric id.
            "id": latest.id,
            "session_id": latest.session_id,
            "status": latest.status,
            "progress": latest.progress,
            "current_epoch": latest.current_epoch,
            "total_epochs": latest.total_epochs,
            "eta_seconds": int(eta.total_seconds()) if eta else None,
            "created_at": latest.created_at,
            "started_at": latest.started_at,
            "completed_at": latest.completed_at,
            "error": (latest.error_message or "")[:300] or None,
        }
    return out


def _models_and_deployment(project) -> Dict[str, Any]:
    from ml_models.models import Model, ModelVersion

    models = Model.objects.filter(project=project, is_deleted=False)
    versions = ModelVersion.objects.filter(model__in=models, is_deleted=False).select_related("model")

    deployed = {"production": [], "staging": []}
    for v in versions.filter(deployment_status__in=deployed.keys()).order_by("-deployed_at"):
        name, value = _metric(v.metrics)
        deployed[v.deployment_status].append({**_version_ref(v), "deployed_at": v.deployed_at,
                                              "metric_name": name, "metric_value": value})

    # Per model: is its newest trained version better than what runs in production?
    candidate = None
    for model in models:
        newest = versions.filter(model=model, status="trained").order_by("-version_number").first()
        if newest is None or newest.deployment_status == "production":
            continue
        production = versions.filter(model=model, deployment_status="production").first()
        name, value = _metric(newest.metrics)
        prod_value = _metric(production.metrics, name)[1] if production is not None and name else None
        delta = value - prod_value if value is not None and prod_value is not None else None
        entry = {
            **_version_ref(newest),
            "metric_name": name,
            "metric_value": value,
            "production_version_name": (production.version_name or f"v{production.version_number}")
            if production is not None else None,
            "production_metric_value": prod_value,
            "delta": delta,
        }
        # Prefer a measured improvement; otherwise a model with nothing in production yet.
        rank = (1, delta) if delta is not None and delta > 0 else (0, 0) if production is None else None
        if rank is not None and (candidate is None or rank > candidate[0]):
            candidate = (rank, entry)

    return {
        "models": {
            "count": models.count(),
            "trained_versions": versions.filter(status="trained").count(),
            "candidate": candidate[1] if candidate else None,
        },
        "deployment": deployed,
    }


def _monitoring(project) -> Dict[str, Any]:
    from temporal_analysis.models import MetricAlert, MetricSnapshot
    from training.models import RetrainingPolicy

    open_alerts = MetricAlert.objects.filter(project=project, is_acknowledged=False)
    by_severity = dict(open_alerts.values_list("severity").annotate(n=Count("id")).values_list("severity", "n"))
    worst = (open_alerts.filter(severity="critical").order_by("-alert_date").first()
             or open_alerts.order_by("-alert_date").first())
    return {
        "open_alerts": sum(by_severity.values()),
        "alerts_by_severity": by_severity,
        "top_alert": None if worst is None else {
            "severity": worst.severity,
            "metric_name": worst.metric_name,
            "message": worst.message,
            "current_value": worst.current_value,
            "threshold_value": worst.threshold_value,
            "alert_date": worst.alert_date,
        },
        "last_snapshot_at": MetricSnapshot.objects.filter(project=project)
        .order_by("-snapshot_date").values_list("snapshot_date", flat=True).first(),
        "active_retraining_policies": RetrainingPolicy.objects.filter(
            model__project=project, model__is_deleted=False, is_active=True
        ).count(),
    }


def build_overview(project) -> Dict[str, Any]:
    """Every stage of one project's pipeline (synchronous; also used by the organization home)."""
    stages = _data_and_annotation(project)
    reviewed_not_in_version = stages.pop("_reviewed_not_in_version")
    return {
        "project": {"project_id": str(project.project_id), "name": project.name},
        "generated_at": timezone.now(),
        **stages,
        "dataset": _dataset(project, reviewed_not_in_version),
        "training": _training(project),
        **_models_and_deployment(project),
        "monitoring": _monitoring(project),
    }


@router.get("/{project_id}/overview", summary="Project pipeline overview")
async def project_overview(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    """Every stage of the project's pipeline in one response (anyone who can open the project)."""
    return await sync_to_async(build_overview)(project_ctx.project)
