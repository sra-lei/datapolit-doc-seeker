"""
docs-seeker - RAG 使用统计（按用户维度，Redis 持久化）

独立于语义缓存（SEMANTIC_CACHE_ENABLED 开关不影响统计）：
- 复用 redis_client 单例，键统一使用 rag:usage:* 前缀
- 记录端：中间件对 /v1/chat、/v1/retrieve 请求做轻量埋点（用户、成功与否）；
  chat_service 对问题文本做热门计数（rag:usage:top ZSet，精确匹配归并）
- 查询端：/v1/usage/stats 聚合（总次数/成功率/活跃用户/用户 Top）；
  /v1/usage/top 返回热门问题 TopN（含语义缓存命中标记，供预热器与 ChatWidget 欢迎语）
- Redis 不可用时静默降级：埋点跳过、查询返回空结构，不影响主流程
"""
import re

from loguru import logger

from docs_seeker.infra.cache.redis_client import get_redis_client

_PREFIX = "rag:usage"

# 需要统计的 RAG 接口
_TRACKED_PATHS = {"/v1/chat", "/v1/retrieve"}

_ANONYMOUS = "anonymous"


class UsageTracker:
    """RAG 使用统计：记录 + 聚合"""

    @staticmethod
    def _key(*parts: str) -> str:
        return ":".join((_PREFIX, *parts))

    def record(self, user_id: str, path: str, status: int) -> None:
        """记录一次 RAG 请求

        Args:
            user_id: 请求方用户 id（X-User-ID 头），空则记为 anonymous
            path: 请求路径（仅 /v1/chat、/v1/retrieve 会被记录）
            status: HTTP 状态码（2xx/3xx 记为成功）
        """
        if path not in _TRACKED_PATHS:
            return
        uid = (user_id or _ANONYMOUS)[:64]
        ok = 200 <= status < 400
        try:
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
        except Exception as e:
            logger.warning(f"[Usage] 记录失败（统计降级）: {e}")

    @staticmethod
    def _normalize_question(q: str) -> str:
        """问题归一化：去首尾空白、压缩连续空白、转小写（精确匹配归并粒度）"""
        if not q:
            return ""
        q = q.strip()
        q = re.sub(r"\s+", " ", q)
        return q.lower()

    def record_question(self, question: str) -> None:
        """记录一次 chat 问题（热门问题 TopN 计数，精确匹配归并）

        仅记录归一化后 2~200 字符的问题；Redis 不可用时静默降级。
        """
        q = self._normalize_question(question)
        if len(q) < 2 or len(q) > 200:
            return
        try:
            redis = get_redis_client()
            redis.zincrby(self._key("top"), 1, q)
        except Exception as e:
            logger.warning(f"[Usage] 记录问题失败（统计降级）: {e}")

    def top_questions(self, limit: int = 10) -> list[dict]:
        """热门问题 TopN：按提问次数降序，附语义缓存命中标记

        Returns:
            [{"question": str, "count": int, "cached": bool}, ...]
        """
        try:
            redis = get_redis_client()
            items = redis.zrevrange(self._key("top"), 0, max(limit - 1, 0), withscores=True)
            # 函数内导入：避免中间件链路加载缓存模块（embedder/OpenAI）
            from docs_seeker.infra.cache.semantic_cache import get_semantic_cache
            cache = get_semantic_cache()

            result = []
            for member, score in items:
                q = str(member)
                if not q:
                    continue
                cached = cache.search(q) is not None
                result.append({"question": q, "count": int(score), "cached": cached})
            return result
        except Exception as e:
            logger.warning(f"[Usage] 热门问题查询失败（降级）: {e}")
            return []

    def stats(self, top_n: int = 20) -> dict:
        """聚合统计

        Returns:
            {
              "total_calls": int, "success_calls": int, "success_rate": str,
              "active_users": int,
              "users": [{"user_id", "calls", "success_rate"}, ...] 按调用次数降序
            }
        """
        empty = {
            "total_calls": 0,
            "success_calls": 0,
            "success_rate": "0.0%",
            "active_users": 0,
            "users": [],
        }
        try:
            redis = get_redis_client()
            total = int(redis.get(self._key("total")) or 0)
            success = int(redis.get(self._key("success")) or 0)
            users = redis.smembers(self._key("users")) or set()

            user_list = []
            for uid in users:
                uid_str = str(uid)
                ut = int(redis.get(self._key("user", uid_str, "total")) or 0)
                us = int(redis.get(self._key("user", uid_str, "success")) or 0)
                user_list.append({
                    "user_id": uid_str,
                    "calls": ut,
                    "success_rate": f"{us / ut:.1%}" if ut else "0.0%",
                })
            user_list.sort(key=lambda x: x["calls"], reverse=True)

            return {
                "total_calls": total,
                "success_calls": success,
                "success_rate": f"{success / total:.1%}" if total else "0.0%",
                "active_users": len(user_list),
                "users": user_list[:top_n],
            }
        except Exception as e:
            logger.warning(f"[Usage] 聚合统计失败（降级）: {e}")
            return empty


_usage_tracker: UsageTracker | None = None


def get_usage_tracker() -> UsageTracker:
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker()
    return _usage_tracker
