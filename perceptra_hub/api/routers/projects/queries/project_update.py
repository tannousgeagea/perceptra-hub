"""
Update a project's editable details.

    PATCH /api/v1/projects/{project_id}

Only the fields present in the body are changed. Project owners/admins (and
org admins/owners) may edit — including inactive projects, so they can be
reactivated. Names are unique per organization → 409.
"""
from uuid import UUID

from asgiref.sync import sync_to_async
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import ProjectContext
from projects.dependencies import get_project_context_including_inactive
from api.routers.projects.schemas import ProjectListItem, ProjectUpdate
from api.routers.projects.queries.project_details import fetch_project_details
from projects.models import Visibility

router = APIRouter(prefix="/projects")

EDITABLE_FIELDS = ("name", "description", "thumbnail_url", "is_active")


@router.patch(
    "/{project_id}",
    response_model=ProjectListItem,
    summary="Update Project",
)
async def update_project(
    project_id: UUID,
    payload: ProjectUpdate,
    # Inactive projects must stay editable, or they could never be reactivated.
    project_ctx: ProjectContext = Depends(get_project_context_including_inactive),
):
    project_ctx.require_project_role("owner", "admin")
    changes = payload.model_dump(exclude_unset=True)

    @sync_to_async
    def apply():
        project = project_ctx.project
        update_fields = []

        for field in EDITABLE_FIELDS:
            if field in changes:
                value = changes[field]
                if field == "name":
                    value = (value or "").strip()
                    if not value:
                        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Project name cannot be empty")
                if field == "thumbnail_url" and value == "":
                    value = None
                setattr(project, field, value)
                update_fields.append(field)

        if changes.get("visibility_id") is not None:
            visibility = Visibility.objects.filter(id=changes["visibility_id"]).first()
            if visibility is None:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Unknown visibility")
            project.visibility = visibility
            update_fields.append("visibility")

        if not update_fields:
            return project

        project.updated_by = project_ctx.user
        update_fields += ["updated_by", "updated_at", "last_edited"]
        try:
            with transaction.atomic():
                project.save(update_fields=update_fields)
        except (IntegrityError, ValidationError) as exc:
            # Project.save() runs full_clean(), so the per-org unique name
            # usually surfaces as a ValidationError before the DB sees it.
            if isinstance(exc, IntegrityError) or "__all__" in getattr(exc, "message_dict", {}):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f'A project named "{project.name}" already exists in this organization',
                )
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, exc.message_dict)
        return project

    project = await apply()
    return await fetch_project_details(project, user=project_ctx.user)
