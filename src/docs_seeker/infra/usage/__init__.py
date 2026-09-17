"""docs-seeker - RAG 使用统计存储（Redis 实现）

业务口径（成功判定 / 归一化 / 聚合）在 domain 的 ``UsageTracker``（domain/services/usage.py），
本模块只提供 Redis 存储原语 ``RedisUsageStore`` 及其单例工厂。
"""

from docs_seeker.infra.usage.redis_store import RedisUsageStore, get_usage_store

__all__ = ["RedisUsageStore", "get_usage_store"]
