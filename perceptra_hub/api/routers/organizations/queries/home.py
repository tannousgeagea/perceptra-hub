"""
The organization home: every project the caller can open, at a glance.

    GET /api/v1/organizations/home

For each project (most recently active first, at most PROJECT_LIMIT) the
same stage data as /projects/{id}/overview, so the console reads both with
one set of rules; plus what is running and what is deployed across them.
Projects follow the Projects page's rule: org admins and owners see all,
everyone else the projects they're a member of.
"""
from typing import Any, Dict, List

from asgiref.sync import sync_to_async
from django.utils import timezone
from fastapi import APIRouter, Depends

from api.dependencies import RequestContext, get_request_context
from api.routers.projects.queries.project_overview import RUNNING_TRAINING, build_overview
from memberships.roles import OrgRole

router = APIRouter()

# Each project's overview is a few dozen queries; past this, page the list.
PROJECT_LIMIT = 30


def _visible_projects(ctx: RequestContext):
    from projects.models import Project

    projects = Project.objects.filter(organization=ctx.organization, is_deleted=False, is_active=True)
    if not (ctx.role and ctx.role.name in (OrgRole.ADMIN, OrgRole.OWNER)):
        projects = projects.filter(memberships__user=ctx.user)
    return Project.objects.filter(pk__in=projects.values("pk"))


def _with_last_activity(projects):
    """
    Annotate last_activity: the latest of project change, image added,
    annotation changed, training started (one query for all projects).
    """
    from annotations.models import Annotation
    from django.db.models import DateTimeField, OuterRef, Subquery
    from django.db.models.functions import Greatest, Coalesce
    from projects.models import ProjectImage
    from training.models import TrainingSession

    def latest(qs, field):
        # qs is ordered newest first; keep that ordering for the [:1].
        return Subquery(qs.values(field)[:1], output_field=DateTimeField())

    last_image = latest(ProjectImage.objects.filter(project=OuterRef("pk")).order_by("-added_at"), "added_at")
    last_annotation = latest(
        Annotation.objects.filter(project_image__project=OuterRef("pk")).order_by("-updated_at"), "updated_at")
    last_training = latest(
        TrainingSession.objects.filter(model_version__model__project=OuterRef("pk")).order_by("-created_at"),
        "created_at")
    # Greatest() is NULL if any argument is: fall back to updated_at for missing ones.
    return projects.annotate(last_activity=Greatest(
        "updated_at",
        Coalesce(last_image, "updated_at"),
        Coalesce(last_annotation, "updated_at"),
        Coalesce(last_training, "updated_at"),
    ))


@sync_to_async
def _home(ctx: RequestContext) -> Dict[str, Any]:
    from training.models import TrainingSession

    visible = _visible_projects(ctx)
    total = visible.count()

    entries: List[Dict[str, Any]] = []
    for project in _with_last_activity(visible).order_by("-last_activity")[:PROJECT_LIMIT]:
        overview = build_overview(project)
        overview["project"]["last_activity_at"] = project.last_activity
        entries.append(overview)

    running = []
    for s in (
        TrainingSession.objects.filter(
            model_version__model__project__in=visible, model_version__model__is_deleted=False,
            status__in=RUNNING_TRAINING,
        )
        .select_related("model_version__model__project")
        .order_by("-created_at")[:20]
    ):
        v = s.model_version
        eta = s.estimated_time_remaining
        running.append({
            "id": s.id,
            "status": s.status,
            "progress": s.progress,
            "current_epoch": s.current_epoch,
            "total_epochs": s.total_epochs,
            "eta_seconds": int(eta.total_seconds()) if eta else None,
            "model_id": v.model.model_id,
            "model_name": v.model.name,
            "version_name": v.version_name or f"v{v.version_number}",
            "project_id": str(v.model.project.project_id),
            "project_name": v.model.project.name,
        })

    production = [
        {**d, "project_id": o["project"]["project_id"], "project_name": o["project"]["name"]}
        for o in entries for d in o["deployment"]["production"]
    ]

    return {
        "organization": {"org_id": str(ctx.organization.org_id), "name": ctx.organization.name},
        "generated_at": timezone.now(),
        "project_count": total,
        "project_limit": PROJECT_LIMIT,
        "projects": entries,
        "running": running,
        "production": production,
    }


@router.get("/organizations/home", summary="Organization home")
async def organization_home(ctx: RequestContext = Depends(get_request_context)):
    """Every project the caller can open, with its pipeline state; running and deployed models."""
    return await _home(ctx)
