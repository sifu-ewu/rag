"""
Professional Caching Layer for RAG System
"""

import json
import hashlib
import time
import logging
from typing import Any, Optional, Dict, Union, List
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Redis for distributed caching (optional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# In-memory caching
import threading
from collections import OrderedDict

from config import config

class CacheBackend(ABC):
    """Abstract cache backend interface"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

class MemoryCache(CacheBackend):
    """In-memory LRU cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.ttl_data = {}
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _is_expired(self, key: str) -> bool:
        """Check if a key has expired"""
        if key not in self.ttl_data:
            return False
        return time.time() > self.ttl_data[key]
    
    def _evict_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self.ttl_data.items()
            if current_time > expiry
        ]
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            del self.ttl_data[key]
            self.stats['evictions'] += 1
    
    def _evict_lru(self):
        """Remove least recently used items"""
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            if oldest_key in self.ttl_data:
                del self.ttl_data[oldest_key]
            self.stats['evictions'] += 1
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            self._evict_expired()
            
            if key in self.cache and not self._is_expired(key):
                # Move to end (mark as recently used)
                value = self.cache[key]
                del self.cache[key]
                self.cache[key] = value
                self.stats['hits'] += 1
                return value
            
            self.stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        with self.lock:
            try:
                self._evict_expired()
                self._evict_lru()
                
                self.cache[key] = value
                
                # Set TTL
                if ttl is None:
                    ttl = self.default_ttl
                
                if ttl > 0:
                    self.ttl_data[key] = time.time() + ttl
                
                self.stats['sets'] += 1
                return True
                
            except Exception as e:
                self.logger.error(f"Cache set error: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            deleted = False
            if key in self.cache:
                del self.cache[key]
                deleted = True
            
            if key in self.ttl_data:
                del self.ttl_data[key]
                deleted = True
            
            if deleted:
                self.stats['deletes'] += 1
            
            return deleted
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.ttl_data.clear()
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'backend': 'memory',
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate_percent': round(hit_rate, 2),
                **self.stats
            }

class RedisCache(CacheBackend):
    """Redis-based distributed cache"""
    
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        
        # Test connection
        try:
            self.redis_client.ping()
            self.logger.info("Redis cache connected successfully")
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            value = self.redis_client.get(key)
            if value is not None:
                return json.loads(value)
            return None
        except Exception as e:
            self.logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in Redis cache"""
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            serialized_value = json.dumps(value, default=str)
            if ttl > 0:
                return self.redis_client.setex(key, ttl, serialized_value)
            else:
                return self.redis_client.set(key, serialized_value)
        except Exception as e:
            self.logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            self.logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            self.redis_client.flushdb()
            return True
        except Exception as e:
            self.logger.error(f"Redis clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        try:
            info = self.redis_client.info()
            return {
                'backend': 'redis',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', 'unknown'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_commands_processed': info.get('total_commands_processed', 0)
            }
        except Exception as e:
            self.logger.error(f"Redis stats error: {e}")
            return {'backend': 'redis', 'error': str(e)}

class CacheManager:
    """Professional cache manager with multiple backends"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.backend = self._initialize_backend()
        self.key_prefix = "rag_cache:"
        
    def _initialize_backend(self) -> CacheBackend:
        """Initialize the appropriate cache backend"""
        if not config.ENABLE_CACHING:
            self.logger.info("Caching disabled")
            return MemoryCache(max_size=0)  # Disabled cache
        
        # Try Redis first if configured
        if config.REDIS_URL and REDIS_AVAILABLE:
            try:
                return RedisCache(config.REDIS_URL, config.CACHE_TTL)
            except Exception as e:
                self.logger.warning(f"Redis cache failed, falling back to memory: {e}")
        
        # Fall back to memory cache
        return MemoryCache(default_ttl=config.CACHE_TTL)
    
    def _make_key(self, key: str) -> str:
        """Create a properly prefixed cache key"""
        return f"{self.key_prefix}{key}"
    
    def _hash_key(self, data: Union[str, Dict, List]) -> str:
        """Create a hash key from data"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True, default=str)
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get_query_result(self, query: str, language: str = None) -> Optional[Dict]:
        """Get cached query result"""
        cache_key = self._make_key(f"query:{self._hash_key(query)}:{language or 'auto'}")
        return await self.backend.get(cache_key)
    
    async def set_query_result(self, query: str, result: Dict, language: str = None, ttl: int = None) -> bool:
        """Cache query result"""
        cache_key = self._make_key(f"query:{self._hash_key(query)}:{language or 'auto'}")
        
        # Add cache metadata
        cached_result = {
            'result': result,
            'cached_at': datetime.now().isoformat(),
            'query_hash': self._hash_key(query)
        }
        
        return await self.backend.set(cache_key, cached_result, ttl)
    
    async def get_document_chunks(self, document_id: str) -> Optional[List[Dict]]:
        """Get cached document chunks"""
        cache_key = self._make_key(f"document_chunks:{document_id}")
        return await self.backend.get(cache_key)
    
    async def set_document_chunks(self, document_id: str, chunks: List[Dict], ttl: int = None) -> bool:
        """Cache document chunks"""
        cache_key = self._make_key(f"document_chunks:{document_id}")
        return await self.backend.set(cache_key, chunks, ttl or 86400)  # 24 hours default
    
    async def get_embeddings(self, text: str) -> Optional[List[float]]:
        """Get cached embeddings"""
        cache_key = self._make_key(f"embeddings:{self._hash_key(text)}")
        return await self.backend.get(cache_key)
    
    async def set_embeddings(self, text: str, embeddings: List[float], ttl: int = None) -> bool:
        """Cache embeddings"""
        cache_key = self._make_key(f"embeddings:{self._hash_key(text)}")
        return await self.backend.set(cache_key, embeddings, ttl or 604800)  # 7 days default
    
    async def invalidate_document(self, document_id: str) -> bool:
        """Invalidate all cache entries for a document"""
        # In a more sophisticated system, you'd track all related keys
        cache_key = self._make_key(f"document_chunks:{document_id}")
        return await self.backend.delete(cache_key)
    
    async def clear_all(self) -> bool:
        """Clear all cache entries"""
        return await self.backend.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.backend.get_stats()

# Global cache manager instance
cache_manager = CacheManager()

def cache_key(func_name: str, *args, **kwargs) -> str:
    """Generate a cache key for function calls"""
    key_data = {
        'function': func_name,
        'args': args,
        'kwargs': kwargs
    }
    return hashlib.sha256(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()

def cached(ttl: int = None, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key_str = f"{key_prefix}:{cache_key(func.__name__, *args, **kwargs)}"
            
            # Try to get from cache
            cached_result = await cache_manager.backend.get(cache_key_str)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await cache_manager.backend.set(cache_key_str, result, ttl)
            
            return result
        return wrapper
    return decorator
