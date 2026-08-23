"""docs-seeker - /v1/stats 路由（只读运行指标，供前端看板展示）"""
from fastapi import APIRouter

from docs_seeker.api.schemas import CacheStats, LLMStats, StatsResponse
from docs_seeker.infra.cache.semantic_cache import get_semantic_cache
from docs_seeker.infra.llm.gateway import get_llm_gateway

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
async def stats():
    """聚合语义缓存 + LLM 网关的运行指标"""
    cache_stats = get_semantic_cache().stats
    llm_stats = get_llm_gateway().stats
    return StatsResponse(
        cache=CacheStats(**cache_stats),
        llm=LLMStats(**llm_stats),
    )
