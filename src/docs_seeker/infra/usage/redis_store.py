"""docs-seeker - 使用统计的 Redis 存储实现

实现 ``UsageStore`` 的业务操作，内部处理键结构（``rag:usage:*`` 前缀）与 Redis
命令（pipeline / ZSet / Set / 计数器）。业务口径（成功判定 / 归一化 / 聚合格式）
在 domain 的 ``UsageTracker``。Redis 不可用时抛异常，由上层静默降级。
"""

from docs_seeker.infra.cache.redis_client import get_redis_client
from docs_seeker.interfaces.usage import CallStats, UsageStore, UserCalls

_PREFIX = "rag:usage"


class RedisUsageStore(UsageStore):
    """基于 Redis 的 UsageStore 实现（复用 redis_client 单例）"""

    @staticmethod
    def _key(*parts: str) -> str:
        return ":".join((_PREFIX, *parts))

    def record_call(self, uid: str, ok: bool) -> None:
        redis = get_redis_client()
        pipe = redis.pipeline()
        pipe.incr(self._key("total"))
        if ok:
            pipe.incr(self._key("success"))
        pipe.incr(self._key("user", uid, "total"))
        if ok:
            pipe.incr(self._key("user", uid, "success"))
        pipe.sadd(self._key("users"), uid)
        pipe.execute()

    def record_question(self, question: str) -> None:
        get_redis_client().zincrby(self._key("top"), 1, question)

    def top_questions(self, limit: int) -> list[tuple[str, int]]:
        # withscores=True 时返回 [(member, score), ...]
        items = get_redis_client().zrevrange(self._key("top"), 0, max(limit - 1, 0), withscores=True)
        return [(str(member), int(score)) for member, score in items]

    def call_stats(self) -> CallStats:
        redis = get_redis_client()
        total = int(redis.get(self._key("total")) or 0)
        success = int(redis.get(self._key("success")) or 0)
        users = redis.smembers(self._key("users")) or set()
        user_list: list[UserCalls] = []
        for uid in users:
            uid_str = str(uid)
            user_list.append(
                UserCalls(
                    user_id=uid_str,
                    total=int(redis.get(self._key("user", uid_str, "total")) or 0),
                    success=int(redis.get(self._key("user", uid_str, "success")) or 0),
                )
            )
        return CallStats(total=total, success=success, users=user_list)


_usage_store: RedisUsageStore | None = None


def get_usage_store() -> RedisUsageStore:
    global _usage_store
    if _usage_store is None:
        _usage_store = RedisUsageStore()
    return _usage_store
