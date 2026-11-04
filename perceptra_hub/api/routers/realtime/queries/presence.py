"""
Who is on a project right now, and where.

PUT    /api/v1/projects/{project_id}/presence   heartbeat (every ~20s and on navigation)
DELETE /api/v1/projects/{project_id}/presence   leaving (tab closed / project left)

Both publish the fresh snapshot to the project's event stream. The user's
name comes from the authenticated user, never from the client.
"""
from typing import Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.dependencies import ProjectContext, get_project_context
from realtime.bus import presence_remove, presence_upsert, publish_async

router = APIRouter()


class PresenceIn(BaseModel):
    client_id: str = Field(..., min_length=8, max_length=64)
    view: Literal["annotate", "project"] = "project"
    # Image open in the annotator (the image uid used in its URL) and its job.
    image_uid: Optional[str] = Field(None, max_length=255)
    job_id: Optional[str] = Field(None, max_length=64)
    # Project page (dataset, annotate, …) when not in the annotator.
    page: Optional[str] = Field(None, max_length=40)


def _display_name(user) -> str:
    full = f"{user.first_name or ''} {user.last_name or ''}".strip()
    return full or user.username or (user.email or "").split("@")[0] or "Someone"


def _require_user(project_ctx: ProjectContext):
    if project_ctx.user is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Presence needs a signed-in user")
    return project_ctx.user


@router.put("/projects/{project_id}/presence", summary="Presence heartbeat")
async def heartbeat(
    project_id: UUID,
    body: PresenceIn,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    user = _require_user(project_ctx)
    project_uuid = str(project_ctx.project.project_id)
    entry = {
        "user_id": str(user.id),
        "name": _display_name(user),
        "view": body.view,
        "image_uid": body.image_uid if body.view == "annotate" else None,
        "job_id": body.job_id,
        "page": body.page,
        "session": body.client_id[-8:],  # tells two tabs of one user apart, nothing more
    }
    snapshot = await presence_upsert(project_uuid, f"{user.id}:{body.client_id}", entry)
    await publish_async(project_uuid, "presence", snapshot)
    return {"count": len(snapshot)}


@router.delete("/projects/{project_id}/presence", status_code=status.HTTP_204_NO_CONTENT, summary="Leave presence")
async def leave(
    project_id: UUID,
    client_id: str = Query(..., min_length=8, max_length=64),
    project_ctx: ProjectContext = Depends(get_project_context),
):
    user = _require_user(project_ctx)
    project_uuid = str(project_ctx.project.project_id)
    snapshot = await presence_remove(project_uuid, f"{user.id}:{client_id}")
    await publish_async(project_uuid, "presence", snapshot)
