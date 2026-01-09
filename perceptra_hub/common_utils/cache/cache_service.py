"""
Modular Cache Service - Pluggable cache backend support
File: core/services/cache_service.py
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, List
import logging

logger = logging.getLogger(__name__)


# ============= Base Cache Interface =============

class BaseCacheBackend(ABC):
    """Abstract cache backend interface"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> bool:
        """Set key-value with optional timeout (seconds)"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    # List operations (for queues)
    @abstractmethod
    def lpush(self, key: str, *values: Any) -> int:
        """Push to list (left/head)"""
        pass
    
    @abstractmethod
    def rpush(self, key: str, *values: Any) -> int:
        """Push to list (right/tail)"""
        pass
    
    @abstractmethod
    def lpop(self, key: str) -> Optional[Any]:
        """Pop from list (left/head)"""
        pass
    
    @abstractmethod
    def rpop(self, key: str) -> Optional[Any]:
        """Pop from list (right/tail)"""
        pass
    
    @abstractmethod
    def lrange(self, key: str, start: int, end: int) -> List[Any]:
        """Get list range"""
        pass
    
    @abstractmethod
    def llen(self, key: str) -> int:
        """Get list length"""
        pass
    
    # Hash operations
    @abstractmethod
    def hset(self, name: str, key: str, value: Any) -> bool:
        """Set hash field"""
        pass
    
    @abstractmethod
    def hget(self, name: str, key: str) -> Optional[Any]:
        """Get hash field"""
        pass
    
    @abstractmethod
    def hgetall(self, name: str) -> dict:
        """Get all hash fields"""
        pass
    
    @abstractmethod
    def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields"""
        pass
    
    # Utility
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache (use with caution)"""
        pass


# ============= Redis Backend =============

class RedisCacheBackend(BaseCacheBackend):
    """Redis cache backend implementation"""
    
    def __init__(self, host: str, port: int, db: int, password: Optional[str] = None):
        import redis
        
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # Test connection
        try:
            self.client.ping()
            logger.info(f"Redis connected: {host}:{port}/{db}")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> bool:
        try:
            return bool(self.client.set(key, value, ex=timeout))
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    # List operations
    def lpush(self, key: str, *values: Any) -> int:
        try:
            return self.client.lpush(key, *values)
        except Exception as e:
            logger.error(f"Redis lpush error: {e}")
            return 0
    
    def rpush(self, key: str, *values: Any) -> int:
        try:
            return self.client.rpush(key, *values)
        except Exception as e:
            logger.error(f"Redis rpush error: {e}")
            return 0
    
    def lpop(self, key: str) -> Optional[Any]:
        try:
            return self.client.lpop(key)
        except Exception as e:
            logger.error(f"Redis lpop error: {e}")
            return None
    
    def rpop(self, key: str) -> Optional[Any]:
        try:
            return self.client.rpop(key)
        except Exception as e:
            logger.error(f"Redis rpop error: {e}")
            return None
    
    def lrange(self, key: str, start: int, end: int) -> List[Any]:
        try:
            return self.client.lrange(key, start, end)
        except Exception as e:
            logger.error(f"Redis lrange error: {e}")
            return []
    
    def llen(self, key: str) -> int:
        try:
            return self.client.llen(key)
        except Exception as e:
            logger.error(f"Redis llen error: {e}")
            return 0
    
    # Hash operations
    def hset(self, name: str, key: str, value: Any) -> bool:
        try:
            return bool(self.client.hset(name, key, value))
        except Exception as e:
            logger.error(f"Redis hset error: {e}")
            return False
    
    def hget(self, name: str, key: str) -> Optional[Any]:
        try:
            return self.client.hget(name, key)
        except Exception as e:
            logger.error(f"Redis hget error: {e}")
            return None
    
    def hgetall(self, name: str) -> dict:
        try:
            return self.client.hgetall(name)
        except Exception as e:
            logger.error(f"Redis hgetall error: {e}")
            return {}
    
    def hdel(self, name: str, *keys: str) -> int:
        try:
            return self.client.hdel(name, *keys)
        except Exception as e:
            logger.error(f"Redis hdel error: {e}")
            return 0
    
    def clear(self) -> bool:
        try:
            self.client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False


# ============= Memcached Backend (Optional) =============

class MemcachedCacheBackend(BaseCacheBackend):
    """Memcached backend (key-value only, no lists/hashes)"""
    
    def __init__(self, servers: List[str]):
        from pymemcache.client.base import Client
        import json
        
        self.client = Client(
            servers[0] if servers else ('localhost', 11211),
            serializer=lambda k, v: json.dumps(v).encode('utf-8'),
            deserializer=lambda k, v: json.loads(v.decode('utf-8'))
        )
        logger.info(f"Memcached connected: {servers}")
    
    def get(self, key: str) -> Optional[Any]:
        return self.client.get(key)
    
    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> bool:
        return self.client.set(key, value, expire=timeout or 0)
    
    def delete(self, key: str) -> bool:
        return self.client.delete(key)
    
    def exists(self, key: str) -> bool:
        return self.get(key) is not None
    
    # List operations - Not supported, use fallback
    def lpush(self, key: str, *values: Any) -> int:
        raise NotImplementedError("Memcached doesn't support list operations")
    
    def rpush(self, key: str, *values: Any) -> int:
        raise NotImplementedError("Memcached doesn't support list operations")
    
    def lpop(self, key: str) -> Optional[Any]:
        raise NotImplementedError("Memcached doesn't support list operations")
    
    def rpop(self, key: str) -> Optional[Any]:
        raise NotImplementedError("Memcached doesn't support list operations")
    
    def lrange(self, key: str, start: int, end: int) -> List[Any]:
        raise NotImplementedError("Memcached doesn't support list operations")
    
    def llen(self, key: str) -> int:
        raise NotImplementedError("Memcached doesn't support list operations")
    
    # Hash operations - Not supported
    def hset(self, name: str, key: str, value: Any) -> bool:
        raise NotImplementedError("Memcached doesn't support hash operations")
    
    def hget(self, name: str, key: str) -> Optional[Any]:
        raise NotImplementedError("Memcached doesn't support hash operations")
    
    def hgetall(self, name: str) -> dict:
        raise NotImplementedError("Memcached doesn't support hash operations")
    
    def hdel(self, name: str, *keys: str) -> int:
        raise NotImplementedError("Memcached doesn't support hash operations")
    
    def clear(self) -> bool:
        return self.client.flush_all()


# ============= In-Memory Backend (Testing/Development) =============

class InMemoryCacheBackend(BaseCacheBackend):
    """In-memory cache backend for testing"""
    
    def __init__(self):
        self.store = {}  # key-value
        self.lists = {}  # lists
        self.hashes = {}  # hashes
        logger.info("In-memory cache initialized")
    
    def get(self, key: str) -> Optional[Any]:
        return self.store.get(key)
    
    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> bool:
        self.store[key] = value
        # Timeout not implemented in memory backend
        return True
    
    def delete(self, key: str) -> bool:
        return self.store.pop(key, None) is not None
    
    def exists(self, key: str) -> bool:
        return key in self.store
    
    # List operations
    def lpush(self, key: str, *values: Any) -> int:
        if key not in self.lists:
            self.lists[key] = []
        for value in reversed(values):
            self.lists[key].insert(0, value)
        return len(self.lists[key])
    
    def rpush(self, key: str, *values: Any) -> int:
        if key not in self.lists:
            self.lists[key] = []
        self.lists[key].extend(values)
        return len(self.lists[key])
    
    def lpop(self, key: str) -> Optional[Any]:
        if key in self.lists and self.lists[key]:
            return self.lists[key].pop(0)
        return None
    
    def rpop(self, key: str) -> Optional[Any]:
        if key in self.lists and self.lists[key]:
            return self.lists[key].pop()
        return None
    
    def lrange(self, key: str, start: int, end: int) -> List[Any]:
        if key not in self.lists:
            return []
        lst = self.lists[key]
        if end == -1:
            return lst[start:]
        return lst[start:end+1]
    
    def llen(self, key: str) -> int:
        return len(self.lists.get(key, []))
    
    # Hash operations
    def hset(self, name: str, key: str, value: Any) -> bool:
        if name not in self.hashes:
            self.hashes[name] = {}
        self.hashes[name][key] = value
        return True
    
    def hget(self, name: str, key: str) -> Optional[Any]:
        return self.hashes.get(name, {}).get(key)
    
    def hgetall(self, name: str) -> dict:
        return self.hashes.get(name, {}).copy()
    
    def hdel(self, name: str, *keys: str) -> int:
        if name not in self.hashes:
            return 0
        count = 0
        for key in keys:
            if self.hashes[name].pop(key, None) is not None:
                count += 1
        return count
    
    def clear(self) -> bool:
        self.store.clear()
        self.lists.clear()
        self.hashes.clear()
        return True


# ============= Cache Service Factory =============

class CacheService:
    """
    Singleton cache service factory.
    Usage: cache = CacheService.get_backend()
    """
    
    _instance: Optional[BaseCacheBackend] = None
    
    @classmethod
    def initialize(cls, backend: str = 'redis', **config):
        """
        Initialize cache backend.
        
        Args:
            backend: 'redis', 'memcached', 'memory'
            config: Backend-specific configuration
        """
        if backend == 'redis':
            cls._instance = RedisCacheBackend(
                host=config.get('host', 'localhost'),
                port=config.get('port', 6379),
                db=config.get('db', 0),
                password=config.get('password')
            )
        elif backend == 'memcached':
            cls._instance = MemcachedCacheBackend(
                servers=config.get('servers', [('localhost', 11211)])
            )
        elif backend == 'memory':
            cls._instance = InMemoryCacheBackend()
        else:
            raise ValueError(f"Unknown cache backend: {backend}")
        
        logger.info(f"Cache service initialized: {backend}")
    
    @classmethod
    def get_backend(cls) -> BaseCacheBackend:
        """Get cache backend instance"""
        if cls._instance is None:
            raise RuntimeError(
                "Cache service not initialized. "
                "Call CacheService.initialize() first"
            )
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset instance (for testing)"""
        cls._instance = None


# ============= Django Integration =============
# File: core/cache.py

from django.conf import settings

def initialize_cache():
    """Initialize cache service from Django settings"""
    cache_config = getattr(settings, 'CACHE_BACKEND', {})
    
    backend = cache_config.get('backend', 'redis')
    
    if backend == 'redis':
        CacheService.initialize(
            backend='redis',
            host=cache_config.get('host', 'localhost'),
            port=cache_config.get('port', 6379),
            db=cache_config.get('db', 0),
            password=cache_config.get('password')
        )
    elif backend == 'memory':
        CacheService.initialize(backend='memory')
    
    return CacheService.get_backend()


# ============= Settings Configuration =============
# Add to settings.py:

"""
CACHE_BACKEND = {
    'backend': 'redis',  # 'redis', 'memcached', 'memory'
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'db': int(os.getenv('REDIS_DB', 0)),
    'password': os.getenv('REDIS_PASSWORD', None)
}
"""

# ============= Usage Example =============

"""
# In your Django app ready() or settings
from core.cache import initialize_cache
initialize_cache()

# In your code
from core.services.cache_service import CacheService

cache = CacheService.get_backend()

# Simple key-value
cache.set('key', 'value', timeout=3600)
value = cache.get('key')

# Lists (job queues)
cache.lpush('jobs:queue', 'job1', 'job2')
job = cache.rpop('jobs:queue')
all_jobs = cache.lrange('jobs:queue', 0, -1)

# Hashes
cache.hset('user:123', 'name', 'John')
name = cache.hget('user:123', 'name')
user = cache.hgetall('user:123')
"""