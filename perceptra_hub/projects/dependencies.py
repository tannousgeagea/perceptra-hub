# projects/dependencies.py
#
# FastAPI dependency functions for project-level authentication.
# Composes: organizations/dependencies.py + projects/resolution.py + projects/context.py

from uuid import UUID

from fastapi import Depends

from organizations.context import RequestContext
from organizations.dependencies import get_request_context
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
