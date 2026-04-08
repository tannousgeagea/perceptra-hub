# projects/context.py

from typing import Optional
from fastapi import HTTPException, status

from memberships.models import Role
from memberships.roles import OrgRole, ProjectRole, roles_for_permission
from organizations.context import RequestContext
from projects.models import Project


class ProjectContext:
    """
    Project-level context that composes an org-level RequestContext.

    All org-level attributes (user, organization, org_role, api_key, auth_method)
    are delegated to the underlying RequestContext so there is no duplication.
    """

    def __init__(
        self,
        org_ctx: RequestContext,
        project: Project,
        project_role: Optional[Role] = None,
    ):
        self._org_ctx = org_ctx
        self.project = project
        self.project_role = project_role

    # ─── Delegation to org context ────────────────────────────────────────────

    @property
    def user(self):
        return self._org_ctx.user

    @property
    def organization(self):
        return self._org_ctx.organization

    @property
    def org_role(self) -> Optional[Role]:
        return self._org_ctx.role

    @property
    def api_key(self):
        return self._org_ctx.api_key

    @property
    def auth_method(self) -> Optional[str]:
        return self._org_ctx.auth_method

    @property
    def is_api_key_auth(self) -> bool:
        return self._org_ctx.is_api_key_auth

    @property
    def effective_user(self):
        return self._org_ctx.effective_user

    # ─── Org-role helpers ─────────────────────────────────────────────────────

    def has_org_role(self, *role_names: str) -> bool:
        """True if user's org role matches any of the given names."""
        return self.org_role is not None and self.org_role.name in role_names

    def is_org_admin(self) -> bool:
        """True if user is org admin or owner (bypasses all project role checks)."""
        return self.has_org_role(OrgRole.ADMIN, OrgRole.OWNER)

    def require_org_role(self, *role_names: str):
        """Raise 403 if user's org role is not in the given names."""
        if not self.has_org_role(*role_names):
            current = self.org_role.name if self.org_role else "none"
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required organization roles: {', '.join(role_names)}. Your role: {current}",
            )

    # ─── Project-role helpers ─────────────────────────────────────────────────

    def is_project_member(self) -> bool:
        """True if user has an explicit project membership."""
        return self.project_role is not None

    def has_project_role(self, *role_names: str) -> bool:
        """True if user's project role matches any of the given names."""
        return self.project_role is not None and self.project_role.name in role_names

    def require_project_role(self, *role_names: str):
        """
        Raise 403 if user's project role is not in the given names.
        Org admins/owners bypass this check automatically.
        """
        if self.is_org_admin():
            return
        if not self.has_project_role(*role_names):
            current = self.project_role.name if self.project_role else "none"
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required project roles: {', '.join(role_names)}. Your role: {current}",
            )

    # ─── Named permission checks ──────────────────────────────────────────────

    def can_view(self) -> bool:
        """Any project member or org admin can view."""
        return self.is_org_admin() or self.is_project_member()

    def can_annotate(self) -> bool:
        """Annotator role or above (or org admin) can create annotations."""
        return self.is_org_admin() or self.has_project_role(
            *roles_for_permission("annotate", scope="project")
        )

    def can_review(self) -> bool:
        """Reviewer role or above (or org admin) can review annotations."""
        return self.is_org_admin() or self.has_project_role(
            *roles_for_permission("review", scope="project")
        )

    def can_edit(self) -> bool:
        """Editor role or above (or org admin) can edit project settings and data."""
        return self.is_org_admin() or self.has_project_role(
            *roles_for_permission("edit", scope="project")
        )

    def can_manage(self) -> bool:
        """Project admin/owner (or org admin) can manage project members."""
        return self.is_org_admin() or self.has_project_role(
            *roles_for_permission("manage", scope="project")
        )

    # ─── Enforcement methods ──────────────────────────────────────────────────

    def require_project_access(self):
        """Require user to be org admin or any project member."""
        if not self.is_org_admin() and not self.is_project_member():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You must be an organization admin or project member.",
            )

    def require_annotate_permission(self):
        if not self.can_annotate():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Annotation permission required. "
                    f"Must be {ProjectRole.ANNOTATOR} or above, or org admin."
                ),
            )

    def require_review_permission(self):
        if not self.can_review():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Review permission required. "
                    f"Must be {ProjectRole.REVIEWER} or above, or org admin."
                ),
            )

    def require_edit_permission(self):
        if not self.can_edit():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Edit permission required. "
                    f"Must be {ProjectRole.EDITOR} or above, or org admin."
                ),
            )

    def require_manage_permission(self):
        if not self.can_manage():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Manage permission required. "
                    f"Must be project {ProjectRole.ADMIN} or {ProjectRole.OWNER}, or org admin."
                ),
            )

    def __repr__(self):
        org_r = self.org_role.name if self.org_role else "none"
        proj_r = self.project_role.name if self.project_role else "org-admin-bypass"
        return (
            f"ProjectContext(user={self.effective_user}, "
            f"project={self.project.name!r}, "
            f"org_role={org_r}, project_role={proj_r})"
        )
