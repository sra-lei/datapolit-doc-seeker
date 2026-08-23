"""docs-seeker - Redis 基础客户端"""
import redis

from docs_seeker.config import settings

_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """返回全局 Redis 客户端（单例，decode_responses=True）"""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client
