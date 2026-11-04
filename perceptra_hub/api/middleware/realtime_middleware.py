"""
Realtime glue for every request (pure ASGI, so streaming responses pass through):

1. Exposes the caller's tab id (X-Client-ID) to the code handling the
   request, so events it causes are tagged with their origin.
2. After a successful write under /api/v1/projects/<uuid>/<resource>/…,
   publishes a coarse "project.changed" event. That covers bulk endpoints
   that update with queryset.update() and never fire model signals.
"""
import re

from realtime.bus import publish_async
from realtime.context import current_origin

PROJECT_PATH = re.compile(r"^/api/v1/projects/([0-9a-fA-F-]{36})/([A-Za-z0-9_-]+)")
MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
# Writes that don't change images, jobs or classes (or that already publish
# precise events: accepting a suggestion saves annotations, which signal).
SKIP_RESOURCES = {"events", "presence", "suggestions", "analytics", "alerts", "members", "models"}
MAX_CLIENT_ID_LENGTH = 64


class RealtimeMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        origin = None
        for name, value in scope.get("headers") or []:
            if name == b"x-client-id":
                origin = value.decode("latin-1")[:MAX_CLIENT_ID_LENGTH] or None
                break

        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        token = current_origin.set(origin)
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            current_origin.reset(token)

        if scope.get("method") in MUTATING_METHODS and status["code"] < 400:
            match = PROJECT_PATH.match(scope.get("path", ""))
            if match and match.group(2) not in SKIP_RESOURCES:
                await publish_async(
                    match.group(1).lower(),
                    "project.changed",
                    {"resource": match.group(2), "method": scope["method"]},
                    origin=origin,
                )
