# projects/resolution.py
#
# Pure project lookup with access verification.
# Imports only Django models — no FastAPI context classes.

from typing import Optional, Tuple
from uuid import UUID

from fastapi import HTTPException, status
from asgiref.sync import sync_to_async

from django.contrib.auth import get_user_model

from memberships.models import ProjectMembership, Role
from memberships.roles import OrgRole
from organizations.models import Organization
from projects.models import Project

User = get_user_model()


@sync_to_async
def fetch_project_with_access_check(
    user: User,                  # type: ignore
    organization: Organization,
    org_role: Optional[Role],
    project_id: UUID,
) -> Tuple[Project, Optional[Role]]:
    """
    Fetch a project and verify the requesting user has access.

    Access rules:
    - Org admin / owner: automatic access, project_role returned as None.
    - All others: must have a ProjectMembership entry; returns that role.

    Returns:
        (project, project_role)  — project_role is None for org admins.

    Raises:
        HTTPException 404: Project not found in this organization.
        HTTPException 403: User has no access.
    """
    try:
        project = Project.objects.select_related('organization').get(
            project_id=project_id,
            organization=organization,
            is_deleted=False,
            is_active=True,
        )
    except Project.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Org admins and owners get automatic project access
    if org_role and org_role.name in (OrgRole.ADMIN, OrgRole.OWNER):
        return project, None

    # All other users must be explicit project members
    try:
        membership = ProjectMembership.objects.select_related('role').get(
            user=user,
            project=project,
        )
        return project, membership.role
    except ProjectMembership.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. You must be an organization admin or project member.",
        )
