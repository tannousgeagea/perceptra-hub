# organizations/context.py

from typing import Optional, Any
from fastapi import HTTPException, status
from organizations.models import Organization
from django.contrib.auth import get_user_model
from memberships.models import OrganizationMembership, Role
from memberships.roles import OrgRole

User = get_user_model()


class RequestContext:
    """
    Unified request context for both JWT and API key authentication.

    The `role` field is always a Role model instance so that all downstream
    has_role() / require_role() calls work unchanged regardless of auth method.
    """

    def __init__(
        self,
        user: User,             # type: ignore
        organization: Organization,
        membership: Optional[OrganizationMembership],
        role: Optional[Role],
        api_key: Optional[Any] = None,
        auth_method: Optional[str] = 'jwt',
    ):
        self.user = user
        self.organization = organization
        self.membership = membership
        self.role = role
        self.api_key = api_key
        self.auth_method = auth_method

    # ─── Auth method helpers ──────────────────────────────────────────────────

    @property
    def is_api_key_auth(self) -> bool:
        return self.auth_method == 'api_key'

    # ─── User resolution ─────────────────────────────────────────────────────

    @property
    def effective_user(self):
        """
        Always returns a user instance safe to assign to created_by / updated_by.

        Resolution order:
          1. JWT-authenticated user (ctx.user)
          2. The user who owns / created the API key
          3. None — field is nullable, record is still created
        """
        if self.user is not None:
            return self.user
        if self.api_key is not None:
            return self.api_key.owned_by or self.api_key.created_by
        return None

    @property
    def effective_api_key(self):
        return self.api_key if self.auth_method == 'api_key' else None

    # ─── Role helpers ─────────────────────────────────────────────────────────

    @property
    def role_name(self) -> Optional[str]:
        """The current role name as a plain string, or None if no role."""
        return self.role.name if self.role else None

    def has_role(self, *role_names: str) -> bool:
        """Return True if the user's role name matches any of the given names."""
        return self.role is not None and self.role.name in role_names

    def require_role(self, *role_names: str):
        """Raise 403 if the user does not hold any of the given roles."""
        if not self.has_role(*role_names):
            current = self.role_name or "none"
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(role_names)}. Your role: {current}",
            )

    # ─── Convenience role checks ──────────────────────────────────────────────

    def is_admin(self) -> bool:
        """True if user is an org admin or owner."""
        return self.role is not None and self.role.name in (OrgRole.ADMIN, OrgRole.OWNER)

    def is_operator(self) -> bool:
        """True if user can perform operational tasks (operator or above)."""
        return self.role is not None and self.role.name in (
            OrgRole.OPERATOR, OrgRole.EDITOR, OrgRole.ADMIN, OrgRole.OWNER,
        )

    def is_viewer(self) -> bool:
        """True if user has at least viewer-level access."""
        return self.role is not None and self.role.name in (
            OrgRole.VIEWER, OrgRole.OPERATOR, OrgRole.EDITOR,
            OrgRole.ADMIN, OrgRole.OWNER,
        )

    # ─── Identity shortcuts ───────────────────────────────────────────────────

    @property
    def organization_id(self) -> str:
        """Public-facing organization UUID as a string."""
        return str(self.organization.org_id)

    @property
    def user_id(self) -> Optional[str]:
        u = self.effective_user
        return str(u.id) if u else None

    def __repr__(self):
        user_label = self.effective_user.email if self.effective_user else 'anonymous'
        return (
            f"RequestContext(user={user_label}, "
            f"org={self.organization.slug}, "
            f"role={self.role_name}, "
            f"auth={self.auth_method})"
        )
