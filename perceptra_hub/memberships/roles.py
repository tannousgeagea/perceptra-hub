# memberships/roles.py
#
# Single source of truth for role names and the permission matrix.
# No imports from Django apps or FastAPI — pure Python only.
# Every other module in the auth pipeline imports from here.

from enum import Enum
from typing import Dict, FrozenSet


# ─── Role enums ───────────────────────────────────────────────────────────────
#
# Both enums subclass str so that:
#   OrgRole.ADMIN == "admin"  →  True
#   role.name in [OrgRole.ADMIN, OrgRole.OWNER]  →  works without .value

class OrgRole(str, Enum):
    """Organization-level roles, ordered from least to most privileged."""
    VIEWER   = "viewer"
    OPERATOR = "operator"
    EDITOR   = "editor"
    ADMIN    = "admin"
    OWNER    = "owner"


class ProjectRole(str, Enum):
    """Project-level roles, ordered from least to most privileged."""
    VIEWER    = "viewer"
    ANNOTATOR = "annotator"
    REVIEWER  = "reviewer"
    EDITOR    = "editor"
    ADMIN     = "admin"
    OWNER     = "owner"


# ─── Role hierarchy (higher rank = more privilege) ────────────────────────────

ORG_ROLE_RANK: Dict[str, int] = {
    OrgRole.VIEWER:   1,
    OrgRole.OPERATOR: 2,
    OrgRole.EDITOR:   3,
    OrgRole.ADMIN:    4,
    OrgRole.OWNER:    5,
}

PROJECT_ROLE_RANK: Dict[str, int] = {
    ProjectRole.VIEWER:    1,
    ProjectRole.ANNOTATOR: 2,
    ProjectRole.REVIEWER:  3,
    ProjectRole.EDITOR:    4,
    ProjectRole.ADMIN:     5,
    ProjectRole.OWNER:     6,
}


# ─── Permission matrix ────────────────────────────────────────────────────────
#
# Maps a named action to the set of roles allowed to perform it.
# Org admins/owners always bypass project-level checks in the dependency layer,
# so that logic is NOT encoded here — it lives in ProjectContext.

PERMISSION_TO_ORG_ROLES: Dict[str, FrozenSet[str]] = {
    # Read org resources (projects, members, models…)
    "view":    frozenset({OrgRole.VIEWER, OrgRole.OPERATOR, OrgRole.EDITOR,
                          OrgRole.ADMIN, OrgRole.OWNER}),
    # Operational tasks (e.g. trigger training runs, import data)
    "operate": frozenset({OrgRole.OPERATOR, OrgRole.EDITOR,
                          OrgRole.ADMIN, OrgRole.OWNER}),
    # Create / update org resources
    "edit":    frozenset({OrgRole.EDITOR, OrgRole.ADMIN, OrgRole.OWNER}),
    # Manage org members, API keys, settings
    "manage":  frozenset({OrgRole.ADMIN, OrgRole.OWNER}),
    # Transfer / delete the organization
    "own":     frozenset({OrgRole.OWNER}),
}

PERMISSION_TO_PROJECT_ROLES: Dict[str, FrozenSet[str]] = {
    # View images, annotations, model outputs
    "view":     frozenset({ProjectRole.VIEWER, ProjectRole.ANNOTATOR,
                           ProjectRole.REVIEWER, ProjectRole.EDITOR,
                           ProjectRole.ADMIN, ProjectRole.OWNER}),
    # Create / edit own annotations
    "annotate": frozenset({ProjectRole.ANNOTATOR, ProjectRole.REVIEWER,
                           ProjectRole.EDITOR, ProjectRole.ADMIN,
                           ProjectRole.OWNER}),
    # Approve / reject others' annotations
    "review":   frozenset({ProjectRole.REVIEWER, ProjectRole.EDITOR,
                           ProjectRole.ADMIN, ProjectRole.OWNER}),
    # Edit project settings, add/remove images, manage datasets
    "edit":     frozenset({ProjectRole.EDITOR, ProjectRole.ADMIN,
                           ProjectRole.OWNER}),
    # Manage project members and roles
    "manage":   frozenset({ProjectRole.ADMIN, ProjectRole.OWNER}),
    # Delete project, transfer ownership
    "own":      frozenset({ProjectRole.OWNER}),
}


# ─── API key permission → org role mapping ───────────────────────────────────
#
# Used in api_keys/auth.py to derive an effective Role from the key's
# declared permission level when no owned_by user is set.

API_KEY_PERMISSION_TO_ORG_ROLE: Dict[str, str] = {
    "read":  OrgRole.VIEWER,
    "write": OrgRole.EDITOR,
    "admin": OrgRole.ADMIN,
}

# Rank table mirroring ORG_ROLE_RANK but keyed by API key permission string.
# Used to take the minimum of (key rank, member rank) for owned_by keys.
API_KEY_PERMISSION_RANK: Dict[str, int] = {
    "read":  ORG_ROLE_RANK[OrgRole.VIEWER],
    "write": ORG_ROLE_RANK[OrgRole.EDITOR],
    "admin": ORG_ROLE_RANK[OrgRole.ADMIN],
}


# ─── Helper ───────────────────────────────────────────────────────────────────

def roles_for_permission(permission: str, scope: str = "org") -> FrozenSet[str]:
    """
    Return the frozenset of role names that have the named permission.

    Args:
        permission: e.g. "view", "annotate", "review", "edit", "manage", "own"
        scope:      "org" (default) or "project"

    Returns:
        frozenset of role name strings; empty frozenset if unknown permission.
    """
    matrix = (
        PERMISSION_TO_PROJECT_ROLES if scope == "project"
        else PERMISSION_TO_ORG_ROLES
    )
    return matrix.get(permission, frozenset())


def org_role_rank(role_name: str) -> int:
    """Return the numeric rank for an org role name (0 if unknown)."""
    return ORG_ROLE_RANK.get(role_name, 0)


def project_role_rank(role_name: str) -> int:
    """Return the numeric rank for a project role name (0 if unknown)."""
    return PROJECT_ROLE_RANK.get(role_name, 0)
