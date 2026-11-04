"""
GET /api/v1/projects/{project_id}/team-pace

Each project member's recent pace and open queue, for sizing job slices
and suggesting who should take a job (see jobs/pace.py).
"""
from typing import List, Optional
from uuid import UUID

from asgiref.sync import sync_to_async
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import ProjectContext, get_project_context
from jobs.pace import team_pace
from memberships.models import ProjectMembership

router = APIRouter(prefix="/projects")


class MemberPace(BaseModel):
    user_id: str
    username: str
    name: str
    role: str
    images_finished: int
    active_hours: float
    active_days: int
    images_per_hour: Optional[float]           # their own, when there is history
    measured: bool                             # enough history to trust images_per_hour
    effective_images_per_hour: Optional[float]  # own when measured, else the team median
    open_images: int                           # still to do in their active jobs
    hours_to_clear_queue: Optional[float]


class TeamPaceOut(BaseModel):
    window_days: int
    team_median_images_per_hour: Optional[float]
    members: List[MemberPace]


@router.get("/{project_id}/team-pace", response_model=TeamPaceOut, summary="Team Pace and Queues")
async def get_team_pace(
    project_id: UUID,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    project_ctx.require_project_access()

    @sync_to_async
    def compute():
        memberships = list(
            ProjectMembership.objects.filter(project=project_ctx.project)
            .select_related("user", "role")
            .order_by("user__username")
        )
        users = {m.user_id: m for m in memberships if m.user.is_active}
        stats = team_pace(project_ctx.organization, users.keys())
        members = []
        for user_id, m in users.items():
            full = f"{m.user.first_name or ''} {m.user.last_name or ''}".strip()
            members.append(MemberPace(
                user_id=str(user_id),
                username=m.user.username,
                name=full or m.user.username,
                role=m.role.name,
                **stats["members"][user_id],
            ))
        return TeamPaceOut(
            window_days=stats["window_days"],
            team_median_images_per_hour=stats["team_median_images_per_hour"],
            members=members,
        )

    return await compute()
