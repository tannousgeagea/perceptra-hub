"""
Per-member job progress for the current organization (Team page).

The organization comes from the request context (X-Organization-ID), like
every other /organizations/<name> endpoint; only org admins and owners may
read it. Everything is computed with two aggregate queries, scoped to the
organization's active projects, jobs and images.
"""
from collections import defaultdict

from asgiref.sync import sync_to_async
from django.db.models import Count, Q
from fastapi import APIRouter, Depends

from api.dependencies import RequestContext, require_organization_admin
from jobs.models import Job, JobImage
from memberships.models import OrganizationMembership

router = APIRouter()


def _display_name(user) -> str:
    full = f"{user.first_name or ''} {user.last_name or ''}".strip()
    return full or user.username


@sync_to_async
def _progress(organization):
    roles = {
        m.user_id: m.role.name
        for m in OrganizationMembership.objects.filter(
            organization=organization, status="active"
        ).select_related("role")
    }

    jobs = list(
        Job.objects.filter(
            project__organization=organization,
            project__is_deleted=False,
            is_active=True,
            assignee_id__in=roles.keys(),
        )
        .select_related("assignee")
        .order_by("created_at")
    )

    counts = {
        row["job_id"]: row
        for row in JobImage.objects.filter(
            job__in=jobs, is_active=True, project_image__is_active=True
        )
        .values("job_id")
        .annotate(
            total=Count("id"),
            annotated=Count("id", filter=Q(project_image__status="annotated")),
            reviewed=Count("id", filter=Q(project_image__status="reviewed")),
            completed=Count("id", filter=Q(project_image__status="dataset")),
        )
    }

    by_user = defaultdict(list)
    for job in jobs:
        by_user[job.assignee_id].append(job)

    data = []
    for user_id, user_jobs in by_user.items():
        user = user_jobs[0].assignee
        totals = {"total": 0, "annotated": 0, "reviewed": 0, "completed": 0}
        assigned = []
        for job in user_jobs:
            c = counts.get(job.id, {})
            for key in totals:
                totals[key] += c.get(key, 0)
            assigned.append({
                "jobId": str(job.job_id),
                "jobName": job.name,
                "totalImages": c.get("total", 0),
                "annotatedImages": c.get("annotated", 0),
                "reviewedImages": c.get("reviewed", 0),
                "completedImages": c.get("completed", 0),
            })
        role = roles.get(user_id, "")
        data.append({
            "userId": str(user_id),
            "userName": _display_name(user),
            "userRole": role,
            "role": role,
            "avatarUrl": None,
            "totalImages": totals["total"],
            "annotatedImages": totals["annotated"],
            "reviewedImages": totals["reviewed"],
            "completedImages": totals["completed"],
            "progressPercentage": round(totals["completed"] / totals["total"] * 100, 2) if totals["total"] else 0.0,
            "lastUpdated": max(j.updated_at for j in user_jobs).isoformat(),
            "assignedJobs": assigned,
        })
    return data


@router.get("/organizations/progress")
async def fetch_user_progress(ctx: RequestContext = Depends(require_organization_admin())):
    return await _progress(ctx.organization)
