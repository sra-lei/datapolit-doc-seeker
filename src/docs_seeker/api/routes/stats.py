"""docs-seeker - /v1/stats 路由（只读运行指标，供前端看板展示）"""

from fastapi import APIRouter

from docs_seeker.api.schemas import CacheStats, LLMStats, StatsResponse
from docs_seeker.infra.cache.semantic_cache import get_semantic_cache
from docs_seeker.infra.llm.client import get_llm_client

router = APIRouter(tags=["stats"])


# 同步 handler：FastAPI 自动放入线程池执行，避免阻塞事件循环
@router.get("/stats", response_model=StatsResponse)
def stats():
    """聚合语义缓存 + LLM 客户端的运行指标"""
    cache_stats = get_semantic_cache().stats
    llm_stats = get_llm_client().stats
    return StatsResponse(
        cache=CacheStats(**cache_stats),
        llm=LLMStats(**llm_stats),
    )
