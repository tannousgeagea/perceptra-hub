# Auth Pipeline Modularization Plan

## Context

The authentication pipeline for the FastAPI layer is currently monolithic — all logic (JWT decoding, org resolution, role checking, project access, context classes) lives in `api/dependencies.py` (844 lines). A restructuring is underway to split this into app-local modules following single-responsibility. Three files exist but are incomplete or inconsistent. The goal is a fully modular, role-defined auth pipeline with a single source of truth for roles and permissions, and clean backward-compatibility so no router import lines change.

---

## Current State

| File | Status | Issues |
|------|--------|--------|
| `users/auth.py` | Done | Correct canonical implementation |
| `organizations/resolution.py` | Exists — broken | Header alias bug (`"-Organization-ID"` → `"X-Organization-ID"`); variable shadowing in `_get_organization_by_id`/`_get_organization_by_slug` |
| `organizations/context.py` | Exists — wrong | `role` stored as `str`; should be `Role` model instance; `organization_id` uses `.id` (PK) not `.org_id` (UUID) |
| `projects/context.py` | Exists — bugged | `api_key` and `auth_method` accepted in `__init__` but never assigned to `self`; missing `can_review()`/`require_review_permission()` |
| `memberships/roles.py` | Missing | No single source of truth for role names or permission matrix |
| `organizations/dependencies.py` | Missing | |
| `projects/resolution.py` | Missing | |
| `projects/dependencies.py` | Missing | |

---

## Target Module Structure

```
perceptra_hub/
├── memberships/
│   └── roles.py                     ← NEW: role enums, hierarchy, permission matrix
├── users/
│   └── auth.py                      ← EXISTS: canonical, no changes needed
├── organizations/
│   ├── resolution.py                ← FIX: 2 bugs
│   ├── context.py                   ← FIX: role type + org_id + role_name property
│   └── dependencies.py              ← NEW: get_request_context + require_* functions
├── projects/
│   ├── resolution.py                ← NEW: fetch_project_with_access_check
│   ├── context.py                   ← FIX: api_key bug + composition + can_review
│   └── dependencies.py              ← NEW: get_project_context
└── api/
    └── dependencies.py              ← SLIM: thin re-export hub (no logic changes for routers)
```

---

## Role & Permission Design (`memberships/roles.py`)

### Org-level roles (hierarchy: OWNER > ADMIN > EDITOR > OPERATOR > VIEWER)
```python
class OrgRole(str, Enum):
    VIEWER = "viewer"; OPERATOR = "operator"; EDITOR = "editor"
    ADMIN = "admin"; OWNER = "owner"
```

### Project-level roles (hierarchy: OWNER > ADMIN > EDITOR > REVIEWER > ANNOTATOR > VIEWER)
```python
class ProjectRole(str, Enum):
    VIEWER = "viewer"; ANNOTATOR = "annotator"; REVIEWER = "reviewer"
    EDITOR = "editor"; ADMIN = "admin"; OWNER = "owner"
```

Using `str, Enum` so that `OrgRole.ADMIN == "admin"` is `True` — DB string comparisons continue to work.

### Permission matrix

**Org-level:**
| Permission | Roles |
|-----------|-------|
| view | viewer, operator, editor, admin, owner |
| operate | operator, editor, admin, owner |
| edit | editor, admin, owner |
| manage | admin, owner |
| own | owner |

**Project-level:**
| Permission | Roles |
|-----------|-------|
| view | viewer, annotator, reviewer, editor, admin, owner |
| annotate | annotator, reviewer, editor, admin, owner |
| review | reviewer, editor, admin, owner |
| edit | editor, admin, owner |
| manage | admin, owner |
| own | owner |

Also define `ORG_ROLE_RANK` and `PROJECT_ROLE_RANK` dicts (role → int) for hierarchy comparisons, and a `roles_for_permission(permission, scope)` helper.

---

## Implementation Steps (ordered to avoid circular imports)

### Step 1 — `memberships/roles.py` (NEW)
- Pure Python: enums, rank dicts, permission maps, `roles_for_permission()` helper
- Zero imports from project code — safe foundation for everything else

### Step 2 — Fix `organizations/context.py`
- Change `role` type from `str` → `Optional[Role]` (model instance)
- Add `role_name` property: `return self.role.name if self.role else None`
- Update `has_role()`: `self.role is not None and self.role.name in role_names`
- Update `is_admin()`, `is_operator()`, `is_viewer()` to use `self.role.name`
- Fix `organization_id` property: `str(self.organization.org_id)` not `.id`
- Add `is_api_key_auth` property: `return self.auth_method == 'api_key'`
- Import `from memberships.roles import OrgRole`

### Step 3 — Fix `organizations/resolution.py`
- Fix header alias: `alias="X-Organization-ID"` (line 10)
- Fix variable shadowing: rename local var from `Organization` → `org` in both helpers

### Step 4 — Update `api_keys/auth.py` (ONE LINE CHANGE)
- Change `from api.dependencies import RequestContext` → `from organizations.context import RequestContext`
- This breaks the only circular import in the graph

### Step 5 — Create `organizations/dependencies.py`
Move from `api/dependencies.py`:
- `_fetch_organization_membership(user, organization)` — private sync_to_async helper
- `get_request_context(request, authorization, x_api_key, x_organization_id, x_organization_slug)` — main entry point (API Key > JWT priority)
- `require_org_role(*required_roles)` — dependency factory
- `require_org_admin()` — shortcut for `require_org_role(OrgRole.ADMIN, OrgRole.OWNER)`
- `require_permission(permission)` — works with both JWT roles and API key permissions, uses `roles_for_permission()` from `memberships/roles.py`
- `require_scope(scope)` — API key only

Move the `from api_keys.auth import authenticate_with_api_key` to module level (cycle is now resolved).

### Step 6 — Create `projects/resolution.py`
Move `fetch_project_with_access_check(user, organization, org_role, project_id)` from `api/dependencies.py`. Imports only Django models — no FastAPI context imports.

### Step 7 — Fix `projects/context.py`
- Change to composition: `__init__(self, org_ctx: RequestContext, project, project_role=None)`
- Delegate via properties: `user`, `organization`, `org_role` (→ `_org_ctx.role`), `api_key`, `auth_method`, `is_api_key_auth`
- Fix the bug: `self.api_key = api_key` and `self.auth_method = auth_method` (now via composition)
- Add `can_review()`: `is_org_admin() or has_project_role(*roles_for_permission("review", "project"))`
- Add `require_review_permission()`
- Update all permission checks to use `memberships.roles` constants

### Step 8 — Create `projects/dependencies.py`
- Move `get_project_context(project_id, ctx)` from `api/dependencies.py`
- Use composition: `ProjectContext(org_ctx=ctx, project=project, project_role=project_role)`

### Step 9 — Slim `api/dependencies.py` to re-export hub
Replace all moved implementations with imports:
```python
from users.auth import get_current_user, fetch_user_from_db
from organizations.context import RequestContext
from organizations.dependencies import (
    get_request_context,
    require_org_role as require_organization_role,
    require_org_admin as require_organization_admin,
    require_permission,
    require_scope,
)
from projects.context import ProjectContext
from projects.dependencies import get_project_context
```

Keep directly in `api/dependencies.py` (no domain home):
- `PaginationParams` + `get_pagination`
- `get_optional_user`
- `bypass_auth_dev`
- `get_db`
- `get_current_organization` (still used by `storage/queries/storage.py`)
- `fetch_organization_and_verify_membership`, `get_user_organization_role` (legacy callers)

---

## Circular Import Safety (dependency graph leaf → root)

```
1. memberships/roles.py              ← no project imports
2. users/auth.py                     ← only Django
3. organizations/resolution.py      ← only organizations.models
4. organizations/context.py         ← memberships.models, organizations.models, memberships.roles
5. api_keys/auth.py                 ← organizations.context (UPDATED from api.dependencies)
6. organizations/dependencies.py    ← users.auth, organizations.resolution, organizations.context, api_keys.auth
7. projects/resolution.py           ← projects.models, memberships.models, memberships.roles
8. projects/context.py              ← organizations.context, memberships.roles, projects.models
9. projects/dependencies.py         ← organizations.dependencies, projects.resolution, projects.context
10. api/dependencies.py             ← re-exports everything above
```

No cycles exist in this graph.

---

## Bugs Fixed

| File | Bug | Fix |
|------|-----|-----|
| `organizations/resolution.py:10` | `alias="-Organization-ID"` | → `alias="X-Organization-ID"` |
| `organizations/resolution.py:55,72` | Local var `Organization` shadows class | Rename to `org` |
| `organizations/context.py:29` | `role: str` | Change to `Optional[Role]` instance |
| `organizations/context.py:79` | `self.organization.id` (PK) | → `self.organization.org_id` (UUID) |
| `organizations/context.py:54` | `self.role in role_names` | → `self.role.name in role_names` |
| `projects/context.py:22-29` | `api_key`/`auth_method` not assigned to `self` | Fixed via composition pattern |
| `projects/context.py` | Missing `can_review()`, `require_review_permission()` | Add both |
| `api_keys/auth.py:~275` | `from api.dependencies import RequestContext` (cycle) | → `from organizations.context import RequestContext` |

---

## Verification

After implementation:
```bash
# Start the container and run Django checks
docker-compose exec perceptrahub python /perceptra_hub/manage.py check

# Confirm FastAPI starts without import errors
docker-compose logs perceptrahub | grep -E "(ERROR|ImportError|startup complete)"

# Hit a JWT-authenticated endpoint
curl -H "Authorization: Bearer <token>" -H "X-Organization-ID: <uuid>" \
  http://localhost:29082/api/v1/projects/

# Hit an API-key-authenticated endpoint
curl -H "X-API-Key: <key>" http://localhost:29082/api/v1/projects/

# Hit a project-scoped endpoint
curl -H "Authorization: Bearer <token>" -H "X-Organization-ID: <uuid>" \
  http://localhost:29082/api/v1/projects/<project_id>/members/
```

All three should return 200 (or 403/401 based on credentials) without `ImportError` or `AttributeError`.