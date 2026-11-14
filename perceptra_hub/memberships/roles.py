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
    # ── API-key-style permission levels (read / write / admin) ──────────────
    # These mirror APIKey.PERMISSION_CHOICES and are the canonical keys for
    # use with Depends(require_permission("read"/"write"/"admin")).
    "read":    frozenset({OrgRole.VIEWER, OrgRole.OPERATOR, OrgRole.EDITOR,
                          OrgRole.ADMIN, OrgRole.OWNER}),
    "write":   frozenset({OrgRole.EDITOR, OrgRole.ADMIN, OrgRole.OWNER}),
    "admin":   frozenset({OrgRole.ADMIN, OrgRole.OWNER}),

    # ── Semantic / human-readable aliases ────────────────────────────────────
    "view":    frozenset({OrgRole.VIEWER, OrgRole.OPERATOR, OrgRole.EDITOR,
                          OrgRole.ADMIN, OrgRole.OWNER}),
    "operate": frozenset({OrgRole.OPERATOR, OrgRole.EDITOR,
                          OrgRole.ADMIN, OrgRole.OWNER}),
    "edit":    frozenset({OrgRole.EDITOR, OrgRole.ADMIN, OrgRole.OWNER}),
    "manage":  frozenset({OrgRole.ADMIN, OrgRole.OWNER}),
    "own":     frozenset({OrgRole.OWNER}),
    "owner":   frozenset({OrgRole.OWNER}),
}

PERMISSION_TO_PROJECT_ROLES: Dict[str, FrozenSet[str]] = {
    # ── API-key-style permission levels ──────────────────────────────────────
    "read":     frozenset({ProjectRole.VIEWER, ProjectRole.ANNOTATOR,
                           ProjectRole.REVIEWER, ProjectRole.EDITOR,
                           ProjectRole.ADMIN, ProjectRole.OWNER}),
    "write":    frozenset({ProjectRole.EDITOR, ProjectRole.ADMIN,
                           ProjectRole.OWNER}),

    # ── Semantic aliases ──────────────────────────────────────────────────────
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
    "admin":   frozenset({ProjectRole.ADMIN, ProjectRole.OWNER}),
    # Delete project, transfer ownership
    "own":      frozenset({ProjectRole.OWNER}),
    "owner":      frozenset({ProjectRole.OWNER}),
}


# Org roles that pass every project-level check (in every project of the org).
ORG_ROLES_WITH_PROJECT_BYPASS: FrozenSet[str] = frozenset({OrgRole.ADMIN, OrgRole.OWNER})


# ─── Role rows ────────────────────────────────────────────────────────────────
#
# Org and project roles share one Role table. Every name here gets a row after
# `migrate` (see memberships/apps.py), so adding a role is a change to this file only.

ALL_ROLE_NAMES: tuple = tuple(dict.fromkeys(
    [r.value for r in OrgRole] + [r.value for r in ProjectRole]
))

ROLE_DESCRIPTIONS: Dict[str, str] = {
    "viewer":    "Can view images, annotations and results",
    "operator":  "Can run inference and operational tasks",
    "annotator": "Can annotate images and mark them done",
    "reviewer":  "Can review annotations and send images back",
    "editor":    "Can add and change images, datasets and models",
    "admin":     "Can manage members, settings and everything editors can",
    "owner":     "Full control, including deleting and transferring",
}


def project_permissions_for(org_role: str | None, project_role: str | None) -> FrozenSet[str]:
    """
    The project permissions a person has, given their org role and their role
    in the project (None if not a member). Org admins and owners have them all.
    This is the rule ProjectContext enforces and the console displays.
    """
    if org_role in ORG_ROLES_WITH_PROJECT_BYPASS:
        return frozenset(PERMISSION_TO_PROJECT_ROLES)  # every permission name
    if not project_role:
        return frozenset()
    return frozenset(
        perm for perm, roles in PERMISSION_TO_PROJECT_ROLES.items() if project_role in roles
    )


def org_permissions_for(org_role: str | None) -> FrozenSet[str]:
    """The org permissions a person with this org role has."""
    if not org_role:
        return frozenset()
    return frozenset(
        perm for perm, roles in PERMISSION_TO_ORG_ROLES.items() if org_role in roles
    )


def role_catalog() -> dict:
    """Roles, their order and the permission matrix, as served to the console."""
    def roles(enum, rank):
        return [
            {"name": r.value, "rank": rank[r], "description": ROLE_DESCRIPTIONS.get(r.value, "")}
            for r in enum
        ]

    def names(roles_):
        # Plain names ("admin"); str() of these enums gives "OrgRole.ADMIN".
        return sorted(getattr(r, "value", r) for r in roles_)

    def matrix(m):
        return {perm: names(roles_) for perm, roles_ in m.items()}

    return {
        "org_roles": roles(OrgRole, ORG_ROLE_RANK),
        "project_roles": roles(ProjectRole, PROJECT_ROLE_RANK),
        "org_permissions": matrix(PERMISSION_TO_ORG_ROLES),
        "project_permissions": matrix(PERMISSION_TO_PROJECT_ROLES),
        "org_roles_with_project_bypass": names(ORG_ROLES_WITH_PROJECT_BYPASS),
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
