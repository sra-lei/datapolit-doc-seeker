"""docs-seeker - 使用统计的 Redis 存储实现

只做 Redis 原语读写（键统一 ``rag:usage:*`` 前缀），不含业务口径。
业务规则（成功判定 / 问题归一化 / TopN 聚合）在 domain 的 ``UsageTracker``。
Redis 不可用时抛异常，由上层（UsageTracker）静默降级。
"""

from docs_seeker.domain.interfaces.usage import UsageStore
from docs_seeker.infra.cache.redis_client import get_redis_client

_PREFIX = "rag:usage"


class RedisUsageStore(UsageStore):
    """基于 Redis 的 UsageStore 实现（复用 redis_client 单例）"""

    @staticmethod
    def _key(*parts: str) -> str:
        return ":".join((_PREFIX, *parts))

    def incr_total(self) -> None:
        get_redis_client().incr(self._key("total"))

    def incr_success(self) -> None:
        get_redis_client().incr(self._key("success"))

    def incr_user_total(self, uid: str) -> None:
        get_redis_client().incr(self._key("user", uid, "total"))

    def incr_user_success(self, uid: str) -> None:
        get_redis_client().incr(self._key("user", uid, "success"))

    def add_user(self, uid: str) -> None:
        get_redis_client().sadd(self._key("users"), uid)

    def get_total(self) -> int:
        return int(get_redis_client().get(self._key("total")) or 0)

    def get_success(self) -> int:
        return int(get_redis_client().get(self._key("success")) or 0)

    def get_users(self) -> set[str]:
        return {str(u) for u in (get_redis_client().smembers(self._key("users")) or set())}

    def get_user_total(self, uid: str) -> int:
        return int(get_redis_client().get(self._key("user", uid, "total")) or 0)

    def get_user_success(self, uid: str) -> int:
        return int(get_redis_client().get(self._key("user", uid, "success")) or 0)

    def incr_top(self, question: str) -> None:
        get_redis_client().zincrby(self._key("top"), 1, question)

    def top_questions(self, limit: int) -> list[tuple[str, float]]:
        # withscores=True 时返回 [(member, score), ...]
        return [
            (str(member), float(score))
            for member, score in get_redis_client().zrevrange(self._key("top"), 0, max(limit - 1, 0), withscores=True)
        ]


_usage_store: RedisUsageStore | None = None


def get_usage_store() -> RedisUsageStore:
    global _usage_store
    if _usage_store is None:
        _usage_store = RedisUsageStore()
    return _usage_store
