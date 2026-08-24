"""docs-seeker - /v1/usage 路由（RAG 使用统计 + 热门问题 TopN）"""
from fastapi import APIRouter, Query

from docs_seeker.api.schemas import (
    UsageStatsResponse,
    UsageTopQuestion,
    UsageTopResponse,
    UsageUserStat,
)
from docs_seeker.infra.usage_tracker import get_usage_tracker

router = APIRouter(tags=["usage"])


@router.get("/usage/stats", response_model=UsageStatsResponse)
async def usage_stats():
    """RAG 使用统计：总次数/成功率/活跃用户/用户 Top（按调用次数降序）"""
    data = get_usage_tracker().stats()
    return UsageStatsResponse(
        total_calls=data["total_calls"],
        success_calls=data["success_calls"],
        success_rate=data["success_rate"],
        active_users=data["active_users"],
        users=[UsageUserStat(**u) for u in data["users"]],
    )


@router.get("/usage/top", response_model=UsageTopResponse)
async def usage_top(limit: int = Query(10, ge=1, le=50)):
    """热门问题 TopN：按提问次数降序，附语义缓存命中标记（ChatWidget 欢迎语 / 预热器用）"""
    items = get_usage_tracker().top_questions(limit=limit)
    return UsageTopResponse(questions=[UsageTopQuestion(**q) for q in items])
