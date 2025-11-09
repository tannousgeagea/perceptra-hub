

"""
Production-ready analytics and statistics endpoints.

Features:
- Optimized queries with select_related/prefetch_related
- Caching with Redis
- Async support
- Proper error handling
- Response models with validation
"""

import hashlib
import json
import logging
from functools import wraps
from django.core.cache import cache
from api.dependencies import ProjectContext

logger = logging.getLogger(__name__)


def make_cache_key(prefix: str, **kwargs):
    serialized = json.dumps(kwargs, sort_keys=True, default=str)
    hashed = hashlib.md5(serialized.encode()).hexdigest()
    return f"{prefix}:{hashed}"

# ============= Cache Decorator =============

def cache_response(timeout: int = 300):
    """Cache decorator for analytics endpoints."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, project_ctx: ProjectContext = None, **kwargs):
            
            try:
                serialized_kwargs = json.dumps(kwargs, sort_keys=True, default=str)
            except TypeError:
                # fallback if non-serializable types appear
                serialized_kwargs = str(kwargs)
                
            # Generate short hash to keep key length safe
            hash_suffix = hashlib.md5(serialized_kwargs.encode()).hexdigest()

            # Generate cache key
            cache_key = f"analytics:{project_ctx.project.project_id}:{func.__name__}:{hash_suffix}"

            # Try to get from cache
            cached = cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return cached
            
            # Execute function
            result = await func(*args, project_ctx=project_ctx, **kwargs)
            
            # Cache result
            cache.set(cache_key, result, timeout)
            logger.debug(f"Cached result for {cache_key}")
            
            return result
        return wrapper
    return decorator