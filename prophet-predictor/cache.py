"""Caching utilities for improved performance."""
import logging
import json
import time
from typing import Optional, Any, Dict
import redis
from redis.exceptions import RedisError

from config import CacheConfig

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager with fallback to in-memory cache."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client = None
        self.memory_cache = {}  # Fallback in-memory cache
        self.memory_cache_timestamps = {}
        
        if config.enabled:
            self._connect_redis()
    
    def _connect_redis(self):
        """Connect to Redis server."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                socket_timeout=5,
                socket_connect_timeout=5,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except RedisError as e:
            logger.warning(f"Failed to connect to Redis: {e}. Falling back to memory cache.")
            self.redis_client = None
        except Exception as e:
            logger.warning(f"Unexpected error connecting to Redis: {e}. Falling back to memory cache.")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.config.enabled:
            return None
        
        # Try Redis first
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    logger.debug(f"Cache hit (Redis): {key}")
                    return json.loads(value)
            except (RedisError, json.JSONDecodeError) as e:
                logger.warning(f"Redis cache get error: {e}")
        
        # Fallback to memory cache
        if key in self.memory_cache:
            timestamp = self.memory_cache_timestamps.get(key, 0)
            if time.time() - timestamp < self.config.ttl_seconds:
                logger.debug(f"Cache hit (memory): {key}")
                return self.memory_cache[key]
            else:
                # Expired entry
                del self.memory_cache[key]
                del self.memory_cache_timestamps[key]
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.config.enabled:
            return False
        
        ttl = ttl or self.config.ttl_seconds
        
        # Try Redis first
        if self.redis_client:
            try:
                serialized_value = json.dumps(value, default=str)
                self.redis_client.setex(key, ttl, serialized_value)
                logger.debug(f"Cache set (Redis): {key}")
                return True
            except (RedisError, json.JSONEncodeError) as e:
                logger.warning(f"Redis cache set error: {e}")
        
        # Fallback to memory cache
        try:
            self.memory_cache[key] = value
            self.memory_cache_timestamps[key] = time.time()
            logger.debug(f"Cache set (memory): {key}")
            
            # Simple cleanup of expired entries
            self._cleanup_memory_cache()
            return True
        except Exception as e:
            logger.error(f"Memory cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.config.enabled:
            return False
        
        success = False
        
        # Try Redis first
        if self.redis_client:
            try:
                self.redis_client.delete(key)
                success = True
            except RedisError as e:
                logger.warning(f"Redis cache delete error: {e}")
        
        # Fallback to memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
            del self.memory_cache_timestamps[key]
            success = True
        
        return success
    
    def get_model(self, key: str) -> Optional[bytes]:
        """Get serialized model from cache."""
        if not self.config.enabled:
            return None
        
        if self.redis_client:
            try:
                value = self.redis_client.get(f"model:{key}")
                if value:
                    logger.debug(f"Model cache hit: {key}")
                    # Redis returns string, we need bytes for pickle
                    return value.encode('latin1') if isinstance(value, str) else value
            except RedisError as e:
                logger.warning(f"Redis model cache get error: {e}")
        
        return None
    
    def set_model(self, key: str, model_data: bytes) -> bool:
        """Set serialized model in cache."""
        if not self.config.enabled:
            return False
        
        if self.redis_client:
            try:
                # Store as binary data
                self.redis_client.setex(
                    f"model:{key}", 
                    self.config.model_cache_ttl, 
                    model_data.decode('latin1')
                )
                logger.debug(f"Model cache set: {key}")
                return True
            except RedisError as e:
                logger.warning(f"Redis model cache set error: {e}")
        
        return False
    
    def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.memory_cache_timestamps.items():
            if current_time - timestamp > self.config.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            del self.memory_cache_timestamps[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "enabled": self.config.enabled,
            "redis_connected": self.redis_client is not None,
            "memory_cache_size": len(self.memory_cache)
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats["redis_used_memory"] = info.get("used_memory_human", "unknown")
                stats["redis_connected_clients"] = info.get("connected_clients", 0)
            except RedisError:
                stats["redis_info"] = "unavailable"
        
        return stats
