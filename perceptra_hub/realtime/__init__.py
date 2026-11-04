"""
Live project events (server-sent events) and presence.

Writes publish small JSON events to a Redis pub/sub channel per project;
browsers on that project hold one SSE stream (api/routers/realtime) and
refresh what changed. Everything here is best-effort: if Redis is down,
writes still succeed and pages simply stop updating live.
"""
