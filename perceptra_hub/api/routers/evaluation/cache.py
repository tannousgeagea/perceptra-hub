"""
Production-Grade Redis Caching Layer
====================================

Implements intelligent caching with invalidation strategies for evaluation metrics.
Handles cache warming, TTL management, and distributed locking.
"""

from common_utils.cache.cache_service import CacheService
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
    """Cache configuration for evaluation"""
    
    # TTL strategies
    summary_ttl: int = 3600  # 1 hour
    detailed_ttl: int = 1800  # 30 min
    image_list_ttl: int = 600  # 10 min
    
    # Invalidation
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
    def image_list_key(cls, project_id: int, page: int, page_size: int, filters: Dict[str, Any]) -> str:
        filter_str = json.dumps(filters, sort_keys=True)
        filter_hash = hashlib.md5(filter_str.encode()).hexdigest()[:8]
        return f"{cls.NAMESPACE}:{cls.VERSION}:images:project:{project_id}:page:{page}:size:{page_size}:filter:{filter_hash}"
    
    
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
    Async cache manager for evaluation metrics.
    Wraps sync CacheService with async interface.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.cache = CacheService.get_backend()
        self._invalidation_tasks: Dict[int, asyncio.Task] = {}
    
    # ------------------------------------------------------------------------
    # CORE OPERATIONS (Async wrappers over sync cache)
    # ------------------------------------------------------------------------
    
    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
        ttl: int,
    ) -> Any:
        """
        Get from cache or compute and store.
        """
        
        # Try cache first (sync operation in thread pool)
        cached = await asyncio.to_thread(self.cache.get, key)
        if cached:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                # Invalid cache, fall through to compute
                pass
        
        # Cache miss - compute (async)
        value = await compute_fn()
        
        # Store in cache (sync operation in thread pool)
        serialized = json.dumps(value, default=str)
        await asyncio.to_thread(self.cache.set, key, serialized, ttl)
        
        return value
    
    async def set_with_ttl(self, key: str, value: Any, ttl: int):
        """Set value with TTL"""
        serialized = json.dumps(value, default=str)
        await asyncio.to_thread(self.cache.set, key, serialized, ttl)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cached = await asyncio.to_thread(self.cache.get, key)
        if cached:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                return None
        return None
    
    async def delete(self, key: str):
        """Delete key from cache"""
        await asyncio.to_thread(self.cache.delete, key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        return await asyncio.to_thread(self.cache.exists, key)
    
    # ------------------------------------------------------------------------
    # PROJECT-SPECIFIC OPERATIONS
    # ------------------------------------------------------------------------
    
    async def get_summary(
        self,
        project_id: int,
        model_version: Optional[str],
        compute_fn: Callable
    ) -> Dict[str, Any]:
        """Get or compute project summary"""
        key = CacheKeyBuilder.summary_key(project_id, model_version)
        return await self.get_or_compute(key, compute_fn, self.config.summary_ttl)
    
    async def get_class_metrics(
        self,
        project_id: int,
        model_version: Optional[str],
        compute_fn: Callable
    ) -> List[Dict[str, Any]]:
        """Get or compute per-class metrics"""
        key = CacheKeyBuilder.class_metrics_key(project_id, model_version)
        return await self.get_or_compute(key, compute_fn, self.config.summary_ttl)
    
    async def invalidate_project(self, project_id: int):
        """
        Invalidate all cached data for a project.
        Uses debouncing to avoid excessive invalidations.
        """
        
        # Cancel pending invalidation if exists
        if project_id in self._invalidation_tasks:
            self._invalidation_tasks[project_id].cancel()
        
        # Schedule debounced invalidation
        async def _debounced_invalidate():
            await asyncio.sleep(self.config.invalidation_debounce_seconds)
            
            # Delete project-related keys
            # Note: Pattern deletion requires scan_delete method in CacheService
            pattern = f"{CacheKeyBuilder.NAMESPACE}:*:project:{project_id}:*"
            
            # Use scan_delete if available, otherwise delete known keys
            if hasattr(self.cache, 'scan_delete'):
                await asyncio.to_thread(self.cache.scan_delete, pattern)
            else:
                # Fallback: delete known key patterns
                for key_type in ['summary', 'classes', 'images']:
                    key = f"{CacheKeyBuilder.NAMESPACE}:{CacheKeyBuilder.VERSION}:{key_type}:project:{project_id}"
                    await asyncio.to_thread(self.cache.delete, key)
            
            # Record invalidation timestamp
            inv_key = CacheKeyBuilder.invalidation_key(project_id)
            await asyncio.to_thread(
                self.cache.set,
                inv_key,
                datetime.utcnow().isoformat(),
                3600  # Keep for 1 hour
            )
            
            # Remove from tracking
            if project_id in self._invalidation_tasks:
                del self._invalidation_tasks[project_id]
        
        task = asyncio.create_task(_debounced_invalidate())
        self._invalidation_tasks[project_id] = task

_cache_manager: Optional[EvaluationCacheManager] = None


def get_evaluation_cache() -> EvaluationCacheManager:
    """Get singleton cache manager instance"""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = EvaluationCacheManager()
    
    return _cache_manager

