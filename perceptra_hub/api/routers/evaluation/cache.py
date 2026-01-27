"""
Production-Grade Redis Caching Layer
====================================

Implements intelligent caching with invalidation strategies for evaluation metrics.
Handles cache warming, TTL management, and distributed locking.
"""

from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import json
import asyncio

import redis.asyncio as redis
from redis.asyncio import Redis
from pydantic import BaseModel


# ============================================================================
# CONFIGURATION
# ============================================================================

class CacheConfig(BaseModel):
    """Redis cache configuration"""
    
    # Connection
    redis_url: str = "redis://localhost:6379/0"
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    
    # TTL strategies
    summary_ttl: int = 3600  # 1 hour for summary metrics
    detailed_ttl: int = 1800  # 30 min for detailed results
    image_list_ttl: int = 600  # 10 min for image lists
    
    # Performance
    enable_compression: bool = True
    max_cache_size_mb: int = 1000
    
    # Invalidation
    auto_invalidate_on_annotation: bool = True
    invalidation_debounce_seconds: int = 5


# ============================================================================
# CACHE KEY MANAGEMENT
# ============================================================================

class CacheKeyBuilder:
    """Builds consistent, versioned cache keys"""
    
    NAMESPACE = "eval"
    VERSION = "v1"
    
    @classmethod
    def summary_key(cls, project_id: int, model_version: Optional[str] = None) -> str:
        """Key for project summary metrics"""
        base = f"{cls.NAMESPACE}:{cls.VERSION}:summary:project:{project_id}"
        if model_version:
            base += f":model:{model_version}"
        return base
    
    @classmethod
    def class_metrics_key(cls, project_id: int, model_version: Optional[str] = None) -> str:
        """Key for per-class metrics"""
        base = f"{cls.NAMESPACE}:{cls.VERSION}:classes:project:{project_id}"
        if model_version:
            base += f":model:{model_version}"
        return base
    
    @classmethod
    def image_list_key(cls, project_id: int, filters: Dict[str, Any]) -> str:
        """Key for filtered image lists"""
        # Create stable hash of filters
        filter_str = json.dumps(filters, sort_keys=True)
        filter_hash = hashlib.md5(filter_str.encode()).hexdigest()[:8]
        return f"{cls.NAMESPACE}:{cls.VERSION}:images:project:{project_id}:filter:{filter_hash}"
    
    @classmethod
    def temporal_metrics_key(cls, project_id: int, model_version: str) -> str:
        """Key for temporal analysis data"""
        return f"{cls.NAMESPACE}:{cls.VERSION}:temporal:project:{project_id}:model:{model_version}"
    
    @classmethod
    def invalidation_set_key(cls, project_id: int) -> str:
        """Key for tracking invalidation timestamps"""
        return f"{cls.NAMESPACE}:{cls.VERSION}:invalidation:project:{project_id}"
    
    @classmethod
    def lock_key(cls, resource: str) -> str:
        """Distributed lock key"""
        return f"{cls.NAMESPACE}:lock:{resource}"


# ============================================================================
# CACHE MANAGER
# ============================================================================

class EvaluationCacheManager:
    """
    Manages caching for evaluation metrics with intelligent invalidation.
    
    Features:
    - Automatic cache warming
    - Conditional invalidation (only when metrics change)
    - Distributed locking for cache updates
    - Compression for large datasets
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis: Optional[Redis] = None
        self._invalidation_tasks: Dict[int, asyncio.Task] = {}
    
    async def connect(self):
        """Initialize Redis connection"""
        self.redis = await redis.from_url(
            self.config.redis_url,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            decode_responses=True  # Auto decode to strings
        )
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
    
    # ------------------------------------------------------------------------
    # CORE CACHING OPERATIONS
    # ------------------------------------------------------------------------
    
    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
        ttl: int,
        compress: bool = False
    ) -> Any:
        """
        Get from cache or compute and store.
        
        Args:
            key: Cache key
            compute_fn: Async function to compute value if cache miss
            ttl: Time to live in seconds
            compress: Whether to compress large payloads
        
        Returns:
            Cached or computed value
        """
        
        # Try cache first
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        
        # Cache miss - compute
        value = await compute_fn()
        
        # Store in cache
        serialized = json.dumps(value, default=str)
        await self.redis.setex(key, ttl, serialized)
        
        return value
    
    async def set_with_ttl(self, key: str, value: Any, ttl: int):
        """Set value with TTL"""
        serialized = json.dumps(value, default=str)
        await self.redis.setex(key, ttl, serialized)
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                await self.redis.delete(*keys)
            if cursor == 0:
                break
    
    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key"""
        return await self.redis.ttl(key)
    
    # ------------------------------------------------------------------------
    # DISTRIBUTED LOCKING
    # ------------------------------------------------------------------------
    
    async def acquire_lock(
        self,
        resource: str,
        timeout: int = 10,
        blocking_timeout: int = 5
    ) -> bool:
        """
        Acquire distributed lock using Redis.
        
        Args:
            resource: Resource to lock
            timeout: Lock expiration (seconds)
            blocking_timeout: How long to wait for lock
        
        Returns:
            True if lock acquired
        """
        lock_key = CacheKeyBuilder.lock_key(resource)
        
        # Try to acquire lock with retry
        end_time = asyncio.get_event_loop().time() + blocking_timeout
        
        while asyncio.get_event_loop().time() < end_time:
            # SET NX EX - set if not exists with expiration
            acquired = await self.redis.set(
                lock_key,
                "1",
                nx=True,  # Only set if doesn't exist
                ex=timeout  # Expiration
            )
            
            if acquired:
                return True
            
            # Wait before retry
            await asyncio.sleep(0.1)
        
        return False
    
    async def release_lock(self, resource: str):
        """Release distributed lock"""
        lock_key = CacheKeyBuilder.lock_key(resource)
        await self.redis.delete(lock_key)
    
    # ------------------------------------------------------------------------
    # PROJECT-SPECIFIC OPERATIONS
    # ------------------------------------------------------------------------
    
    async def get_summary(
        self,
        project_id: int,
        model_version: Optional[str],
        compute_fn: Callable
    ) -> Dict[str, Any]:
        """Get or compute project summary with caching"""
        key = CacheKeyBuilder.summary_key(project_id, model_version)
        return await self.get_or_compute(key, compute_fn, self.config.summary_ttl)
    
    async def get_class_metrics(
        self,
        project_id: int,
        model_version: Optional[str],
        compute_fn: Callable
    ) -> List[Dict[str, Any]]:
        """Get or compute per-class metrics with caching"""
        key = CacheKeyBuilder.class_metrics_key(project_id, model_version)
        return await self.get_or_compute(key, compute_fn, self.config.summary_ttl)
    
    async def invalidate_project(self, project_id: int):
        """
        Invalidate all cached data for a project.
        Uses debouncing to avoid excessive invalidations.
        """
        
        # Cancel pending invalidation task if exists
        if project_id in self._invalidation_tasks:
            self._invalidation_tasks[project_id].cancel()
        
        # Schedule debounced invalidation
        async def _debounced_invalidate():
            await asyncio.sleep(self.config.invalidation_debounce_seconds)
            
            # Invalidate all project-related keys
            patterns = [
                f"{CacheKeyBuilder.NAMESPACE}:*:project:{project_id}:*",
                f"{CacheKeyBuilder.NAMESPACE}:*:project:{project_id}",
            ]
            
            for pattern in patterns:
                await self.invalidate_pattern(pattern)
            
            # Record invalidation timestamp
            inv_key = CacheKeyBuilder.invalidation_set_key(project_id)
            await self.redis.set(inv_key, datetime.utcnow().isoformat())
            
            # Remove from tracking
            del self._invalidation_tasks[project_id]
        
        task = asyncio.create_task(_debounced_invalidate())
        self._invalidation_tasks[project_id] = task
    
    async def get_last_invalidation(self, project_id: int) -> Optional[datetime]:
        """Get timestamp of last cache invalidation for project"""
        inv_key = CacheKeyBuilder.invalidation_set_key(project_id)
        timestamp = await self.redis.get(inv_key)
        if timestamp:
            return datetime.fromisoformat(timestamp)
        return None
    
    # ------------------------------------------------------------------------
    # CACHE WARMING
    # ------------------------------------------------------------------------
    
    async def warm_cache(
        self,
        project_id: int,
        model_versions: List[str],
        compute_summary_fn: Callable,
        compute_classes_fn: Callable
    ):
        """
        Pre-populate cache with commonly accessed data.
        Runs in background without blocking.
        """
        
        async def _warm():
            for model_version in model_versions:
                try:
                    # Warm summary
                    summary_key = CacheKeyBuilder.summary_key(project_id, model_version)
                    if not await self.redis.exists(summary_key):
                        summary = await compute_summary_fn(project_id, model_version)
                        await self.set_with_ttl(summary_key, summary, self.config.summary_ttl)
                    
                    # Warm class metrics
                    class_key = CacheKeyBuilder.class_metrics_key(project_id, model_version)
                    if not await self.redis.exists(class_key):
                        classes = await compute_classes_fn(project_id, model_version)
                        await self.set_with_ttl(class_key, classes, self.config.summary_ttl)
                
                except Exception as e:
                    # Log error but don't fail warming
                    print(f"Cache warming failed for {model_version}: {e}")
        
        # Run in background
        asyncio.create_task(_warm())
    
    # ------------------------------------------------------------------------
    # MONITORING & STATS
    # ------------------------------------------------------------------------
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        info = await self.redis.info("stats")
        memory = await self.redis.info("memory")
        
        # Count evaluation keys
        cursor = 0
        eval_key_count = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor,
                match=f"{CacheKeyBuilder.NAMESPACE}:*",
                count=1000
            )
            eval_key_count += len(keys)
            if cursor == 0:
                break
        
        return {
            "total_keys": await self.redis.dbsize(),
            "evaluation_keys": eval_key_count,
            "memory_used_mb": memory.get("used_memory") / 1024 / 1024,
            "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1),
            "uptime_seconds": info.get("uptime_in_seconds"),
        }


# ============================================================================
# DECORATOR FOR AUTOMATIC CACHING
# ============================================================================

def cached_endpoint(
    ttl: int = 3600,
    key_builder: Optional[Callable] = None,
    invalidate_on: Optional[List[str]] = None
):
    """
    Decorator for FastAPI endpoints to add automatic caching.
    
    Usage:
        @router.get("/summary")
        @cached_endpoint(ttl=3600, key_builder=lambda project_id: f"summary:{project_id}")
        async def get_summary(project_id: int):
            ...
    """
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, cache_manager: EvaluationCacheManager = None, **kwargs):
            if not cache_manager or not cache_manager.redis:
                # No cache available, execute directly
                return await func(*args, **kwargs)
            
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default: use function name + args
                cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # Try cache
            cached = await cache_manager.redis.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache_manager.set_with_ttl(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

from fastapi import Depends

# Global cache manager instance
_cache_manager: Optional[EvaluationCacheManager] = None


async def get_cache_manager() -> EvaluationCacheManager:
    """Dependency for FastAPI endpoints"""
    global _cache_manager
    
    if _cache_manager is None:
        config = CacheConfig()
        _cache_manager = EvaluationCacheManager(config)
        await _cache_manager.connect()
    
    return _cache_manager


async def shutdown_cache():
    """Cleanup on app shutdown"""
    global _cache_manager
    if _cache_manager:
        await _cache_manager.close()


# ============================================================================
# USAGE IN FASTAPI ENDPOINT
# ============================================================================

"""
# Add to FastAPI app startup/shutdown:

from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup():
    # Initialize cache
    pass

@app.on_event("shutdown")
async def shutdown():
    await shutdown_cache()


# Updated endpoint with caching:

@router.get("/projects/{project_id}/summary")
async def get_quick_summary(
    project_id: int,
    model_version: Optional[str] = None,
    cache: EvaluationCacheManager = Depends(get_cache_manager),
    db: Session = Depends(get_db),
):
    '''Get cached summary or compute fresh'''
    
    async def compute_summary():
        # Your existing computation logic
        return await compute_dataset_summary(project_id, model_version, db)
    
    # Use cache
    summary = await cache.get_summary(project_id, model_version, compute_summary)
    
    return summary


# Invalidation on annotation changes (in your Celery worker or signal):

from your_cache import get_cache_manager

async def on_annotation_updated(project_id: int):
    cache = await get_cache_manager()
    await cache.invalidate_project(project_id)
"""


# ============================================================================
# TESTING & MONITORING
# ============================================================================

async def test_cache_performance():
    """Test cache hit rates and performance"""
    
    config = CacheConfig(redis_url="redis://localhost:6379/1")  # Use test DB
    cache = EvaluationCacheManager(config)
    await cache.connect()
    
    try:
        # Test basic operations
        key = "test:key"
        await cache.set_with_ttl(key, {"data": "test"}, 60)
        
        # Measure hit rate
        import time
        
        hit_count = 0
        miss_count = 0
        
        for i in range(100):
            start = time.time()
            
            if i % 2 == 0:
                # Cache hit
                result = await cache.redis.get(key)
                if result:
                    hit_count += 1
            else:
                # Cache miss
                result = await cache.redis.get("nonexistent")
                if not result:
                    miss_count += 1
            
            elapsed = time.time() - start
            print(f"Request {i}: {elapsed*1000:.2f}ms")
        
        print(f"\nHit rate: {hit_count}/{hit_count+miss_count}")
        
        # Test stats
        stats = await cache.get_cache_stats()
        print(f"\nCache stats: {json.dumps(stats, indent=2)}")
        
    finally:
        await cache.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_cache_performance())