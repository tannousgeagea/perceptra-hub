"""
GET /api/v1/projects/{project_id}/events — server-sent events for one project.

Sends a presence snapshot first, then every event published for the project
(see realtime.bus), and a comment line every 15s so proxies keep the
connection open. The stream ends after STREAM_MAX_SECONDS; the client
reconnects, which re-checks its token and project access.

It also carries the user's own channel (notifications), so a tab needs one
stream in a project and one outside (GET /notifications/events), never two.
"""
import asyncio
import json
import logging
import time
from typing import AsyncIterator, Callable, Optional, Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from api.dependencies import ProjectContext, RequestContext, get_project_context, get_request_context
from realtime.bus import async_client, channel, presence_snapshot, user_channel

logger = logging.getLogger(__name__)
router = APIRouter()

HEARTBEAT_SECONDS = 15
STREAM_MAX_SECONDS = 25 * 60


def _frame(payload: str) -> str:
    return f"data: {payload}\n\n"


def sse_response(
    request: Request,
    channels: Sequence[str],
    label: str,
    intro: Optional[Callable[[], AsyncIterator[str]]] = None,
) -> StreamingResponse:
    """Stream every message published on `channels` as server-sent events."""

    async def stream():
        redis = async_client()
        pubsub = redis.pubsub()
        try:
            await pubsub.subscribe(*channels)
            yield "retry: 3000\n\n"
            if intro is not None:
                async for frame in intro():
                    yield frame

            started = last_sent = time.monotonic()
            while time.monotonic() - started < STREAM_MAX_SECONDS:
                if await request.is_disconnected():
                    break
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message.get("type") == "message":
                    data = message["data"]
                    yield _frame(data.decode() if isinstance(data, bytes) else str(data))
                    last_sent = time.monotonic()
                elif time.monotonic() - last_sent >= HEARTBEAT_SECONDS:
                    yield ": ping\n\n"
                    last_sent = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Redis unavailable: end the stream; the client retries with backoff.
            logger.warning("realtime stream for %s ended: %s", label, exc)
        finally:
            try:
                await pubsub.unsubscribe()
                # redis-py >= 5 has aclose(); older versions only close().
                await (pubsub.aclose() if hasattr(pubsub, "aclose") else pubsub.close())
            except Exception:
                pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # nginx: don't buffer the stream
            "Connection": "keep-alive",
        },
    )


@router.get("/projects/{project_id}/events", summary="Live project events (SSE)")
async def project_events(
    project_id: UUID,
    request: Request,
    project_ctx: ProjectContext = Depends(get_project_context),
):
    project_uuid = str(project_ctx.project.project_id)

    async def presence_first():
        snapshot = await presence_snapshot(project_uuid)
        yield _frame(json.dumps({"type": "presence", "data": snapshot, "ts": time.time()}))

    channels = [channel(project_uuid)]
    if project_ctx.user is not None:
        channels.append(user_channel(project_ctx.user.id))
    return sse_response(request, channels, f"project {project_uuid}", presence_first)


@router.get("/notifications/events", summary="Live notifications (SSE), for pages outside a project")
async def notification_events(
    request: Request,
    ctx: RequestContext = Depends(get_request_context),
):
    if ctx.user is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="A user is required")
    return sse_response(request, [user_channel(ctx.user.id)], f"user {ctx.user.id}")
