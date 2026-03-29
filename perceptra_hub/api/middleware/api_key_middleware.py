"""
Middleware for automatic API key usage logging and rate limit response headers.
"""
import time
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that runs after request processing to:
    1. Log API key usage (endpoint, method, status code, response time, IP, user agent)
    2. Add X-RateLimit-* response headers for API key authenticated requests
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()

        response = await call_next(request)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Check if this request was authenticated with an API key
        api_key = getattr(request.state, 'api_key', None)
        if api_key is None:
            return response

        # Log usage asynchronously
        try:
            await _log_usage(
                api_key=api_key,
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                response_time_ms=elapsed_ms,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get('user-agent'),
            )
        except Exception as e:
            # Never let logging failures break the response
            logger.error(f"Failed to log API key usage: {e}")

        # Add rate limit headers
        response.headers["X-RateLimit-Limit-Minute"] = str(api_key.rate_limit_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(api_key.rate_limit_per_hour)

        return response


@sync_to_async
def _log_usage(
    api_key,
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: int,
    ip_address: str = None,
    user_agent: str = None,
):
    """Create an APIKeyUsageLog entry."""
    from api_keys.models import APIKeyUsageLog

    APIKeyUsageLog.objects.create(
        api_key=api_key,
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        response_time_ms=response_time_ms,
        ip_address=ip_address,
        user_agent=user_agent,
    )
