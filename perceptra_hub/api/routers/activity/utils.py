# apps/activity/cache.py
from django.core.cache import cache
from django.conf import settings
import hashlib
import json


class ActivityCache:
    """Redis caching layer for expensive queries."""
    
    CACHE_TIMEOUTS = {
        'user_summary': 300,  # 5 minutes
        'project_progress': 60,  # 1 minute
        'leaderboard': 600,  # 10 minutes
        'timeline': 120,  # 2 minutes
    }
    
    @staticmethod
    def get_cache_key(prefix: str, **params) -> str:
        """Generate deterministic cache key."""
        param_string = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_string.encode()).hexdigest()[:8]
        return f"activity:{prefix}:{param_hash}"
    
    @classmethod
    def get_or_compute(cls, key_prefix: str, compute_func, **params):
        """Get from cache or compute and store."""
        cache_key = cls.get_cache_key(key_prefix, **params)
        
        # Try cache first
        result = cache.get(cache_key)
        if result is not None:
            return result
        
        # Compute and cache
        result = compute_func(**params)
        timeout = cls.CACHE_TIMEOUTS.get(key_prefix, 300)
        cache.set(cache_key, result, timeout)
        
        return result
    
    @classmethod
    def invalidate(cls, key_prefix: str, **params):
        """Invalidate specific cache entry."""
        cache_key = cls.get_cache_key(key_prefix, **params)
        cache.delete(cache_key)
    
    @classmethod
    def invalidate_pattern(cls, pattern: str):
        """Invalidate all keys matching pattern."""
        # Requires Redis SCAN command
        pass  # Implementation depends on your cache backend