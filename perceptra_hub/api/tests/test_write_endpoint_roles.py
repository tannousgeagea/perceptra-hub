"""
Guard: every data-API route that changes something must check a role.

test_route_auth makes sure each route authenticates its caller. This test goes
one step further for write routes (POST / PUT / PATCH / DELETE): being signed
in, or being a project member, is not enough on its own — the route must also
check a role or permission (memberships/roles.py).

A route passes when its endpoint, its dependencies, or the helper functions
they call in the same module contain a recognisable check (see CHECK_PATTERN).
Routes that need no role on purpose are listed in NO_ROLE_NEEDED, each with
the reason. A new write route without a check fails this test until it gets
one or is added here deliberately.

This catches a missing check, not a wrong one: whether the right role is
required is covered by the role tests for each area.

Run:  python manage.py test api.tests.test_write_endpoint_roles
"""
import inspect
import re
import sys
import types
import unittest

from api.tests.test_route_auth import PUBLIC_ROUTES, _routes  # also runs django.setup()

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# Dependencies that only authenticate (who is calling, which organization, and
# for get_project_context: member or org admin). They prove no role, so their
# source is not searched for a check.
AUTH_ONLY_DEPENDENCIES = {
    "get_current_user",
    "get_request_context",
    "get_project_context",
    "get_project_context_including_inactive",
    "authenticate_agent_from_header",
}

# What counts as a role check: require_*(...) helpers and ProjectContext /
# RequestContext methods, roles_for_permission, API-key permission checks, and
# staff-only maintenance endpoints.
CHECK_PATTERN = re.compile(
    r"require_\w*\)?\("        # ctx.require_edit_permission(), require_project_permission(...),
                               # sync_to_async(require_project_permission)(...)
    r"|\.has_role\("
    r"|has_permission\("
    r"|\.can_\w+\("
    r"|\.is_admin\("
    r"|roles_for_permission\("
    r"|check_permission\("
    r"|\.is_staff\b"
)

# (method, path relative to /api/v1): why no role check is needed.
NO_ROLE_NEEDED = {
    # Your own account
    ("POST", "/auth/logout"): "signs out the caller",
    ("PUT", "/auth/me"): "edits the caller's own profile",
    ("POST", "/auth/change-password"): "changes the caller's own password (needs the current one)",
    ("DELETE", "/auth/oauth/unlink/{provider}"): "unlinks a sign-in provider from the caller's own account",
    # Your own notifications (filtered to recipient = caller)
    ("POST", "/notifications/read-all"): "the caller's own notifications",
    ("POST", "/notifications/{notification_id}/read"): "the caller's own notifications",
    ("DELETE", "/notifications"): "the caller's own notifications",
    # The invitation token is the credential
    ("POST", "/invitations/{token}/accept"): "the unguessable invitation token is the credential",
    # Compute agents: agent key + secret, scoped to the agent's own jobs and organization
    ("POST", "/agents/heartbeat"): "agent credentials; updates the calling agent only",
    ("POST", "/agents/jobs/progress"): "agent credentials; the agent's own jobs only",
    ("POST", "/agents/jobs/complete"): "agent credentials; the agent's own jobs only",
    ("POST", "/agents/artifacts/upload"): "agent credentials; only its own running job's artifact files",
    ("POST", "/edge/heartbeat"): "device key; reports and reads the calling device's own state only",
    ("POST", "/edge/rotate-key"): "device key; replaces the calling device's own key only",
    ("POST", "/edge/samples"): "device key; the model's sampling policy (set by project admins) and the site switch decide",
    ("PUT", "/edge/samples/{sample_id}/file"): "device key; only the calling device's own pending sample",
    ("POST", "/edge/samples/{sample_id}/done"): "device key; only the calling device's own sample, into the model's project",
    ("POST", "/agents/inference-jobs/{job_id}/results"): "agent credentials; the agent's own jobs only",
}


def _source(fn) -> str:
    try:
        return inspect.getsource(inspect.unwrap(fn))
    except (TypeError, OSError):
        return ""


def _with_helpers(fn, depth: int, seen: set) -> str:
    """Source of fn plus the same-module functions it calls, `depth` levels deep."""
    fn = inspect.unwrap(fn)
    if id(fn) in seen:
        return ""
    seen.add(id(fn))
    text = _source(fn)
    module = sys.modules.get(getattr(fn, "__module__", "") or "")
    if depth == 0 or module is None:
        return text
    for name in set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", text)):
        obj = getattr(module, name, None)
        # sync_to_async(...) wraps the function: look through it.
        target = getattr(obj, "__wrapped__", None) or getattr(obj, "func", None) or obj
        if isinstance(target, types.FunctionType) and target.__module__ == module.__name__:
            text += "\n" + _with_helpers(target, depth - 1, seen)
    return text


def _dependency_sources(dependant, seen: set) -> str:
    text = ""
    for sub in dependant.dependencies:
        if getattr(sub.call, "__name__", "") not in AUTH_ONLY_DEPENDENCIES:
            text += "\n" + _with_helpers(sub.call, 1, seen)
        text += _dependency_sources(sub, seen)
    return text


def write_routes_without_role_check():
    from api.main import create_app

    seen, missing = set(), []
    for route in _routes(create_app().routes):
        for method in sorted(route.methods & WRITE_METHODS):
            key = (method, route.path)
            if key in seen:
                continue
            seen.add(key)
            if key in PUBLIC_ROUTES or key in NO_ROLE_NEEDED:
                continue
            visited: set = set()
            text = _with_helpers(route.endpoint, 2, visited) + _dependency_sources(route.dependant, visited)
            if not CHECK_PATTERN.search(text):
                missing.append(f"{method} {route.path}  ({route.endpoint.__module__})")
    return seen, missing


class WriteEndpointRolesTest(unittest.TestCase):
    def test_every_write_route_checks_a_role(self):
        seen, missing = write_routes_without_role_check()
        self.assertGreater(len(seen), 100, "route discovery found too few write routes")
        self.assertEqual(
            missing, [],
            "Write routes without a role check. Add one (see memberships/roles.py), or "
            "list the route in NO_ROLE_NEEDED with the reason:\n  " + "\n  ".join(missing),
        )

    def test_no_role_list_has_no_stale_entries(self):
        seen, _ = write_routes_without_role_check()
        stale = sorted(f"{m} {p}" for (m, p) in NO_ROLE_NEEDED if (m, p) not in seen)
        self.assertEqual(stale, [], "NO_ROLE_NEEDED lists routes that no longer exist")


if __name__ == "__main__":
    unittest.main()
