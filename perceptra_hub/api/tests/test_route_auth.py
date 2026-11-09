"""
Guard: every data-API route must authenticate its caller.

A route passes when its dependency tree reaches one of the auth roots below
(a signed-in user, the organization request context, or agent credentials).
Routes that are public on purpose are listed in PUBLIC_ROUTES, each with the
reason. A new route without auth fails this test until it gets an auth
dependency or is added here deliberately.

Run:  python manage.py test api.tests.test_route_auth
"""
import os
import unittest

import django

if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ["DJANGO_SETTINGS_MODULE"] = "perceptra_hub.settings"
django.setup()

from fastapi.routing import APIRoute  # noqa: E402

# Dependency functions that authenticate the caller. Everything else that
# authenticates (require_permission, get_project_context, ...) is built on one
# of these, so reaching any of them anywhere in the tree is enough.
AUTH_ROOTS = {
    "get_current_user",                 # users.auth (JWT)
    "get_request_context",              # organizations.dependencies (JWT or X-API-Key + org)
    "authenticate_agent_from_header",   # compute agents (X-Agent-Key + X-Agent-Secret)
}

# (method, path relative to /api/v1): why it is public.
PUBLIC_ROUTES = {
    ("POST", "/auth/login"): "sign-in",
    ("POST", "/auth/register"): "sign-up",
    ("POST", "/auth/refresh"): "takes the refresh token in the body",
    ("POST", "/auth/password-reset-request"): "forgot password",
    ("POST", "/auth/password-reset"): "takes the reset token in the body",
    ("POST", "/auth/oauth/initiate"): "starts the OAuth sign-in",
    ("GET", "/auth/oauth/callback"): "OAuth provider redirect, verified by the signed state",
    ("POST", "/auth/oauth/callback"): "OAuth code exchange, verified by the signed state",
    ("GET", "/media/images/{image_id}"): "HMAC-signed, expiring URL (storage.signing)",
    ("GET", "/media/files/{storage_path:path}"): "HMAC-signed, expiring URL (MEDIA_REQUIRE_SIGNED_URLS)",
    ("GET", "/agents/install-agent.sh"): "installer script; the agent authenticates afterwards",
    ("GET", "/project-types/"): "static list of project types",
    ("GET", "/visibility-options/"): "static list of visibility options",
    ("GET", "/health"): "health check",
    ("GET", "/evaluation/health"): "health check",
    ("GET", "/invitations/{token}"): "invitation page; the unguessable token in the link is the credential",
}


def _dependency_names(dependant) -> set:
    names = set()
    for sub in dependant.dependencies:
        names.add(getattr(sub.call, "__name__", type(sub.call).__name__))
        names |= _dependency_names(sub)
    return names


def _routes(routes):
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        for attr in ("routes", "router", "original_router", "app"):
            inner = getattr(route, attr, None)
            inner_routes = getattr(inner, "routes", None) if attr != "routes" else inner
            if inner_routes:
                yield from _routes(inner_routes)
                break


def unauthenticated_routes():
    from api.main import create_app

    seen, missing = set(), []
    for route in _routes(create_app().routes):
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            key = (method, route.path)
            if key in seen:
                continue
            seen.add(key)
            if key in PUBLIC_ROUTES:
                continue
            if not (_dependency_names(route.dependant) & AUTH_ROOTS):
                missing.append(f"{method} {route.path}  ({route.endpoint.__module__})")
    return seen, missing


class RouteAuthTest(unittest.TestCase):
    def test_every_route_authenticates(self):
        seen, missing = unauthenticated_routes()
        self.assertGreater(len(seen), 100, "route discovery found too few routes")
        self.assertEqual(
            missing, [],
            "Routes without authentication. Add an auth dependency, or list the "
            "route in PUBLIC_ROUTES with the reason it is public:\n  " + "\n  ".join(missing),
        )

    def test_public_list_has_no_stale_entries(self):
        seen, _ = unauthenticated_routes()
        stale = sorted(f"{m} {p}" for (m, p) in PUBLIC_ROUTES if (m, p) not in seen)
        self.assertEqual(stale, [], "PUBLIC_ROUTES lists routes that no longer exist")


if __name__ == "__main__":
    unittest.main()
