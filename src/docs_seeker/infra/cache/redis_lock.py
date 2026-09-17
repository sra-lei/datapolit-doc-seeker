"""docs-seeker - 分布式锁的 Redis 实现

基于 Redis ``SET key val NX EX ttl`` 实现互斥，复用 redis_client 单例。
Redis 不可用时**抛异常**（由调用方决定降级策略，见 domain 的 ``TopQuestionWarmup``）。
"""

from docs_seeker.domain.interfaces.lock import DistributedLock
from docs_seeker.infra.cache.redis_client import get_redis_client


class RedisDistributedLock(DistributedLock):
    """基于 Redis 的分布式锁实现"""

    def acquire(self, key: str, ttl_seconds: int) -> bool:
        return bool(get_redis_client().set(key, "1", nx=True, ex=ttl_seconds))

    def release(self, key: str) -> None:
        get_redis_client().delete(key)


_redis_lock: RedisDistributedLock | None = None


def get_distributed_lock() -> RedisDistributedLock:
    global _redis_lock
    if _redis_lock is None:
        _redis_lock = RedisDistributedLock()
    return _redis_lock
