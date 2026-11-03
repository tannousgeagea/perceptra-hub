"""
Publishing project events and keeping presence, over Redis.

Channel per project: mtx:rt:project:<project uuid>
Event (JSON): {"type", "data", "actor" (user id or null), "origin" (tab id or null), "ts"}
"""
import json
import logging
import threading
import time
from typing import Any, Optional

import redis
import redis.asyncio as aioredis
from django.conf import settings
from django.db import transaction

from realtime.context import current_origin

logger = logging.getLogger(__name__)

CHANNEL_PREFIX = "mtx:rt:project:"
PRESENCE_PREFIX = "mtx:rt:presence:"
# Clients heartbeat every 20s; an entry older than this is gone.
PRESENCE_STALE_SECONDS = 50
PRESENCE_KEY_TTL_SECONDS = 3600

_UNSET = object()


def channel(project_uuid) -> str:
    return f"{CHANNEL_PREFIX}{project_uuid}"


def presence_key(project_uuid) -> str:
    return f"{PRESENCE_PREFIX}{project_uuid}"


def redis_url() -> str:
    return f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"


def encode_event(event_type: str, data: dict, *, actor: Any = None, origin: Optional[str] = None) -> str:
    return json.dumps(
        {"type": event_type, "data": data, "actor": actor, "origin": origin, "ts": time.time()},
        default=str,
    )


# ─── Sync side (Django ORM code, signals, Celery) ────────────────────────────

_sync_client: Optional[redis.Redis] = None
_sync_lock = threading.Lock()


def _client() -> redis.Redis:
    global _sync_client
    if _sync_client is None:
        with _sync_lock:
            if _sync_client is None:
                _sync_client = redis.Redis.from_url(redis_url(), socket_connect_timeout=2, socket_timeout=2)
    return _sync_client


def publish(project_uuid, event_type: str, data: dict, *, actor: Any = None, origin: Any = _UNSET) -> None:
    """Publish now. Never raises: live updates must not break writes."""
    if project_uuid is None:
        return
    if origin is _UNSET:
        origin = current_origin.get()
    try:
        _client().publish(channel(project_uuid), encode_event(event_type, data, actor=actor, origin=origin))
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("realtime publish failed (%s): %s", event_type, exc)


def publish_on_commit(project_uuid, event_type: str, data: dict, *, actor: Any = None) -> None:
    """Publish once the current transaction commits (immediately outside one)."""
    origin = current_origin.get()  # capture now, the callback may run later
    transaction.on_commit(lambda: publish(project_uuid, event_type, data, actor=actor, origin=origin))


# ─── Async side (FastAPI: middleware, SSE stream, presence) ──────────────────

_async_pool: Optional[aioredis.ConnectionPool] = None


def async_client() -> aioredis.Redis:
    """Shared pool per worker process (one event loop per uvicorn worker)."""
    global _async_pool
    if _async_pool is None:
        _async_pool = aioredis.ConnectionPool.from_url(redis_url(), socket_connect_timeout=2)
    return aioredis.Redis(connection_pool=_async_pool)


async def publish_async(project_uuid, event_type: str, data: dict, *, actor: Any = None, origin: Optional[str] = None) -> None:
    try:
        await async_client().publish(channel(project_uuid), encode_event(event_type, data, actor=actor, origin=origin))
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("realtime publish failed (%s): %s", event_type, exc)


async def presence_snapshot(project_uuid) -> list:
    """Live presence entries of a project; stale ones are dropped on the way."""
    r = async_client()
    key = presence_key(project_uuid)
    raw = await r.hgetall(key)
    now = time.time()
    alive, stale = [], []
    for field, value in raw.items():
        try:
            entry = json.loads(value)
        except (TypeError, ValueError):
            stale.append(field)
            continue
        if now - float(entry.get("ts", 0)) > PRESENCE_STALE_SECONDS:
            stale.append(field)
        else:
            alive.append(entry)
    if stale:
        await r.hdel(key, *stale)
    return alive


async def presence_upsert(project_uuid, field: str, entry: dict) -> list:
    r = async_client()
    key = presence_key(project_uuid)
    entry = {**entry, "ts": time.time()}
    async with r.pipeline(transaction=False) as pipe:
        pipe.hset(key, field, json.dumps(entry, default=str))
        pipe.expire(key, PRESENCE_KEY_TTL_SECONDS)
        await pipe.execute()
    return await presence_snapshot(project_uuid)


async def presence_remove(project_uuid, field: str) -> list:
    await async_client().hdel(presence_key(project_uuid), field)
    return await presence_snapshot(project_uuid)
