from contextvars import ContextVar
from typing import Optional

# The browser tab (X-Client-ID header) whose request is being handled.
# Events carry it as "origin" so that tab can skip its own changes.
# Set by api.middleware.realtime_middleware; asgiref's sync_to_async copies
# the context, so ORM code (and its signals) sees it too.
current_origin: ContextVar[Optional[str]] = ContextVar("realtime_origin", default=None)
