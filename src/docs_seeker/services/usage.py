"""
docs-seeker - RAG 使用统计（领域服务）

业务口径全部在此：哪些请求算使用（_TRACKED_PATHS）、什么算成功（2xx/3xx）、
问题怎么归一化（精确匹配归并粒度）、TopN 与聚合怎么算。持久化交给注入的
``UsageStore``（infra 提供 Redis 实现），缓存命中判断交给注入的
``SemanticCachePort`` —— domain 不感知 Redis。

Redis 不可用时静默降级：记录跳过、查询返回空结构，不影响主流程。
"""

import re
from typing import Any

from loguru import logger

from docs_seeker.interfaces.cache import SemanticCachePort
from docs_seeker.interfaces.usage import UsageStore

# 需要统计的 RAG 接口
_TRACKED_PATHS = {"/v1/chat"}

_ANONYMOUS = "anonymous"


class UsageTracker:
    """RAG 使用统计：记录 + 聚合（业务规则层，存储走 UsageStore）"""

    def __init__(self, store: UsageStore, cache: SemanticCachePort):
        self._store = store
        self._cache = cache

    def record(self, user_id: str, path: str, status: int) -> None:
        """记录一次 RAG 请求

        Args:
            user_id: 请求方用户 id（X-User-ID 头），空则记为 anonymous
            path: 请求路径（仅 /v1/chat 会被记录）
            status: HTTP 状态码（2xx/3xx 记为成功）
        """
        if path not in _TRACKED_PATHS:
            return
        uid = (user_id or _ANONYMOUS)[:64]
        ok = 200 <= status < 400
        try:
            self._store.record_call(uid, ok)
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

        仅记录归一化后 2~200 字符的问题；存储不可用时静默降级。
        """
        q = self._normalize_question(question)
        if len(q) < 2 or len(q) > 200:
            return
        try:
            self._store.record_question(q)
        except Exception as e:
            logger.warning(f"[Usage] 记录问题失败（统计降级）: {e}")

    def top_questions(self, limit: int = 10) -> list[dict]:
        """热门问题 TopN：按提问次数降序，附语义缓存命中标记

        Returns:
            [{"question": str, "count": int, "cached": bool}, ...]
        """
        try:
            items = self._store.top_questions(limit)
            result = []
            for member, score in items:
                q = str(member)
                if not q:
                    continue
                cached = self._cache.search(q) is not None
                result.append({"question": q, "count": score, "cached": cached})
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
            stats = self._store.call_stats()
            user_list: list[dict[str, Any]] = [
                {
                    "user_id": u.user_id,
                    "calls": u.total,
                    "success_rate": f"{u.success / u.total:.1%}" if u.total else "0.0%",
                }
                for u in stats.users
            ]
            user_list.sort(key=lambda x: x["calls"], reverse=True)

            return {
                "total_calls": stats.total,
                "success_calls": stats.success,
                "success_rate": f"{stats.success / stats.total:.1%}" if stats.total else "0.0%",
                "active_users": len(user_list),
                "users": user_list[:top_n],
            }
        except Exception as e:
            logger.warning(f"[Usage] 聚合统计失败（降级）: {e}")
            return empty
