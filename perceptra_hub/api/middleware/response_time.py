"""
X-Response-Time on every HTTP response: seconds from the request reaching the
app to the response headers being sent (CORS already exposes the header).

Plain ASGI rather than BaseHTTPMiddleware, so streamed responses (downloads,
event streams) pass through untouched.
"""
import time

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class ResponseTimeMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()

        async def send_with_timing(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Response-Time"] = f"{time.perf_counter() - started:.6f}"
            await send(message)

        await self.app(scope, receive, send_with_timing)
