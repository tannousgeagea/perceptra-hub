# projects/dependencies.py
#
# FastAPI dependency functions for project-level authentication.
# Composes: organizations/dependencies.py + projects/resolution.py + projects/context.py

from typing import Optional
from uuid import UUID

from fastapi import Depends

from organizations.context import RequestContext
from organizations.dependencies import get_request_context, require_permission
from projects.context import ProjectContext
from projects.resolution import fetch_project_with_access_check


async def get_project_context(
    project_id: UUID,
    ctx: RequestContext = Depends(get_request_context),
) -> ProjectContext:
    """
    Resolve the full project context for a request.

    Automatically verifies project access:
    - Org admin / owner → automatic access (project_role will be None).
    - Project member → access granted with their project role.
    - Everyone else → 403 Forbidden.

    Usage:
        @router.get("/projects/{project_id}/images")
        async def list_images(
            project_ctx: ProjectContext = Depends(get_project_context),
        ):
            project_ctx.require_project_access()
            ...

        @router.post("/projects/{project_id}/annotations")
        async def create_annotation(
            project_ctx: ProjectContext = Depends(get_project_context),
        ):
            project_ctx.require_annotate_permission()
            ...
    """
    project, project_role = await fetch_project_with_access_check(
        ctx.user,
        ctx.organization,
        ctx.role,
        project_id,
    )

    return ProjectContext(
        org_ctx=ctx,
        project=project,
        project_role=project_role,
    )


async def require_image_write(ctx: RequestContext, project_id: Optional[UUID]):
    """
    Adding images (upload, link from storage): the datalake needs "write" in the
    organization; a project needs "write" in that project (project editor+, or
    org admin), the same rule as every other project action.

    API keys must also hold the "write" key permission. Returns the project, or
    None for the datalake. Raises 403/404.
    """
    if project_id is None or ctx.api_key:
        await require_permission("write")(ctx=ctx)
    if project_id is None:
        return None
    project, project_role = await fetch_project_with_access_check(
        ctx.user, ctx.organization, ctx.role, project_id
    )
    ProjectContext(org_ctx=ctx, project=project, project_role=project_role).require_edit_permission()
    return project


_ROLE_FOR_PERMISSION = {
    "annotate": "annotator", "review": "reviewer", "edit": "editor",
    "write": "editor", "manage": "admin", "admin": "admin",
}


def require_project_permission(ctx: RequestContext, project, permission: str, what: str = "This action") -> None:
    """
    Raise 403 unless the person has `permission` in `project` — for objects that
    belong to a project but are reached by their own id (models, versions…).
    Same rule as ProjectContext (memberships.roles.project_permissions_for):
    org admins and owners pass everywhere. Synchronous: call it from sync code
    or through sync_to_async.
    """
    from memberships.models import ProjectMembership
    from memberships.roles import project_permissions_for

    project_role = None
    if project is not None and ctx.user is not None:
        membership = (
            ProjectMembership.objects.filter(user=ctx.user, project=project)
            .select_related("role")
            .first()
        )
        project_role = membership.role.name if membership else None
    allowed = project_permissions_for(ctx.role.name if ctx.role else None, project_role)
    if permission not in allowed:
        from fastapi import HTTPException, status
        role = _ROLE_FOR_PERMISSION.get(permission, permission)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{what} needs the {role} role or above in the project",
        )


async def get_project_context_including_inactive(
    project_id: UUID,
    ctx: RequestContext = Depends(get_request_context),
) -> ProjectContext:
    """
    Like get_project_context, but also resolves inactive (archived) projects.
    Only for endpoints that manage the project itself — e.g. reactivating it.
    """
    project, project_role = await fetch_project_with_access_check(
        ctx.user,
        ctx.organization,
        ctx.role,
        project_id,
        include_inactive=True,
    )
    return ProjectContext(org_ctx=ctx, project=project, project_role=project_role)
